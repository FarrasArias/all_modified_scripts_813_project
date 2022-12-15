#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <set>

#include <google/protobuf/util/json_util.h>

#include "lz4.h"
#include "../protobuf/midi.pb.h"
#include "../encoder/encoder_all.h"
#include "../enum/encoder_types.h"
#include "../enum/train_config.h"
#include "../random.h"

// START OF NAMESPACE
namespace mmm {

template<class T>
using matrix = std::vector<std::vector<T>>;

template<class T>
using tensor = std::vector<std::vector<std::vector<T>>>;

template <class T>
class Batcher {
public:
  Batcher( int mmaxlen, std::mt19937 *e) {
    maxlen = mmaxlen;
    batch_maxlen = 0;
    batch_size = 0;
    engine = e;
  }
  void add( std::vector<T> &seq ) {
    std::vector<T> item;
    if (seq.size() > maxlen) {
      int off = random_on_range((int)seq.size() - maxlen + 1, engine);
      copy(seq.begin() + off, seq.begin() + off + maxlen, back_inserter(item));
    }
    else {
      copy(seq.begin(), seq.end(), back_inserter(item));
    }
    batch.push_back( item );
    batch_maxlen = std::max(item.size(), batch_maxlen);
    batch_size++;
  }
  void pad( T value ) {
    for (size_t i=0; i<batch_size; i++) {
      batch[i].insert(batch[i].end(), batch_maxlen-batch[i].size(), value);
    }
  }
  std::mt19937 *engine;
  size_t batch_size;
  size_t maxlen;
  size_t batch_maxlen;
  std::vector<std::vector<T>> batch;
};


class Jagged {
public:
  Jagged(std::string filepath_) {
    filepath = filepath_;
    header_filepath = filepath_ + ".header";
    can_write = false;
    can_read = false;
    flush_count = 0;
    num_bars = 4;
    min_tracks = 2;
    max_tracks = 12;
    max_seq_len = 1024;

    engine.seed(time(NULL));

    encoder = NULL;
  }

  void set_seed(int seed) {
    srand(seed); // set the seed
    engine.seed(seed);
  }

  void set_num_bars(int x) {
    num_bars = x;
  }

  void set_min_tracks(int x) {
    min_tracks = x;
  }

  void set_max_tracks(int x) {
    max_tracks = x;
  }

  void set_max_seq_len(int x) {
    max_seq_len = x;
  }

  void enable_write() {
    assert(can_read == false);
    if (can_write) { return; }
    // check that the current file is empty unless force flag is present ?
    fs.open(filepath, std::ios::out | std::ios::binary);
    can_write = true;
  }
  
  void enable_read() {
    assert(can_write == false);
    if (can_read) { return; }
    fs.open(filepath, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
      throw std::runtime_error("COULD NOT OPEN FILE!");
    }
    header_fs.open(header_filepath, std::ios::in | std::ios::binary);
    header.ParseFromIstream(&header_fs);
    can_read = true;
  }

  void append(std::string &s, size_t split_id) {
    enable_write();

    size_t start = fs.tellp();
    // begin compress ===============================
    size_t src_size = sizeof(char)*s.size();
    size_t dst_capacity = LZ4_compressBound(src_size);
    char* dst = new char[dst_capacity];
    size_t dst_size = LZ4_compress_default(
      (char*)s.c_str(), dst, src_size, dst_capacity);
    fs.write(dst, dst_size);
    delete[] dst;
    // end compress =================================
    size_t end = fs.tellp();
    midi::Dataset::Item *item;
    switch (split_id) {
      case 0: item = header.add_train(); break;
      case 1: item = header.add_valid(); break;
      case 2: item = header.add_test(); break;
    }
    item->set_start(start);
    item->set_end(end);
    item->set_src_size(src_size);
    flush_count++;

    if (flush_count >= 1000) {
      flush();
      flush_count = 0;
    }
  }

  std::string read(size_t index, size_t split_id) {
    enable_read();

    midi::Dataset::Item item;
    switch (split_id) {
      case 0: item = header.train(index); break;
      case 1: item = header.valid(index); break;
      case 2: item = header.test(index); break;
    }
    size_t csize = item.end() - item.start();
    char* src = new char[csize/sizeof(char)];
    fs.seekg(item.start());
    fs.read(src, csize);
    std::string x(item.src_size(), ' ');
    LZ4_decompress_safe(src,(char*)x.c_str(),csize,item.src_size());
    delete[] src;
    return x;
  }

  py::bytes read_bytes(size_t index, size_t split_id) {
    return py::bytes(read(index, split_id));
  }

  std::string read_json(size_t index, size_t split_id) {
    midi::Piece p;
    std::string serialized_data = read(index, split_id);
    p.ParseFromString(serialized_data);
    std::string json_string;
    google::protobuf::util::MessageToJsonString(p, &json_string);
    return json_string;
  }

  // below is functions for dataset
  int select_random_transpose(midi::Piece *p) {
    std::tuple<int,int> pitch_ext = get_pitch_extents(p);
    std::vector<int> choices;
    for (int tr=-6; tr<6; tr++) {
      if ((std::get<0>(pitch_ext)+tr >= 0) && (std::get<1>(pitch_ext)+tr < 128)) {
        choices.push_back( tr );
      }
    }
    return choices[random_on_range(choices.size(),&engine)];
  }

  void load_random_piece(midi::Piece *p, size_t split_id) {
    int nitems = get_split_size(split_id);
    int index = random_on_range(nitems, &engine);
    std::string serialized_data = read(index, split_id);
    p->ParseFromString(serialized_data);
  }

  void load_random_segment(midi::Piece *p, size_t split_id, ENCODER *enc, TrainConfig *tc) {

    load_random_piece(p, split_id);
    select_random_segment(
      p, tc->num_bars, tc->min_tracks, tc->max_tracks, 
      enc->config->te, &engine);
    enc->config->transpose = select_random_transpose(p);

    // 75 % of the time we do bar infill
    if (enc->config->both_in_one) {
      enc->config->do_multi_fill = random_on_unit(&engine) < .75;
    }

    // pick bars for infilling if needed
    if (enc->config->do_multi_fill) {
      enc->config->multi_fill = make_bar_mask(
        p, tc->max_mask_percentage, &engine);
    }
  }

  std::vector<int> load_piece(size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {
    midi::Piece p;
    load_random_piece(&p, split_id);

    std::unique_ptr<ENCODER> enc = getEncoder(et);
    if (!enc) {
      throw std::runtime_error("ENCODER TYPE DOES NOT EXIST");
    }

    select_random_segment(&p, tc->num_bars, tc->min_tracks, tc->max_tracks, enc->config->te, &engine);

    enc->config->transpose = select_random_transpose(&p);
    return enc->encode(&p);
  }

  std::tuple<std::vector<int>,std::vector<int>> load_piece_pair(size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {

    while (true) {
      try {
        std::unique_ptr<ENCODER> enc = getEncoder(et);
        if (!enc) {
          throw std::runtime_error("ENCODER TYPE DOES NOT EXIST");
        }

        midi::Piece p;
        load_random_piece(&p, split_id);

        int start;
        std::vector<int> valid_tracks;
        select_random_segment_indices(&p, tc->num_bars*2, tc->min_tracks, tc->max_tracks, enc->config->te, &engine, valid_tracks, &start);

        midi::Piece a(p);
        std::vector<int> abars = arange(start,start+tc->num_bars,1);
        prune_tracks_dev2(&a, valid_tracks, abars);

        shuffle(valid_tracks.begin(), valid_tracks.end(), engine);
        if ((valid_tracks.size()>1) && (random_on_unit(&engine)<.5)) {
          valid_tracks.resize((int)valid_tracks.size() - 1);
        }

        midi::Piece b(p);
        std::vector<int> bbars = arange(
          start+tc->num_bars,start+tc->num_bars*2,1);
        prune_tracks_dev2(&b, valid_tracks, bbars);

        enc->config->transpose = select_random_transpose(&p);
        return make_tuple(enc->encode(&a), enc->encode(&b));
      }
      catch (const std::exception &exc)
      {
        //std::cerr << "ERROR ENCODING" << std::endl;
      }
    }
  }

  std::tuple<matrix<int>,matrix<int>> load_piece_pair_batch(int batch_size, size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {
    enable_read();
    matrix<int> a;
    matrix<int> b;
    for (int i=0; i<batch_size; i++) {
      auto ab = load_piece_pair(split_id, et, tc);
      a.push_back( std::get<0>(ab) );
      b.push_back( std::get<1>(ab) );
    }
    return make_tuple(a,b);
  }

  std::tuple<matrix<int>,matrix<int>> read_batch_v2(int batch_size, size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {
    enable_read();
    std::unique_ptr<ENCODER> enc = getEncoder(et);
    if (!enc) {
      throw std::runtime_error("ENCODER TYPE DOES NOT EXIST");
    }

    Batcher<int> batch(max_seq_len, &engine);
    Batcher<int> att_mask(max_seq_len, &engine);

    // switch number of bars
    std::vector<int> num_bar_choices;
    if (enc->rep->has_token_type(NUM_BARS)) {
      auto it = enc->rep->token_domains.find(NUM_BARS);
      for (const auto v : it->second.input_domain) {
        num_bar_choices.push_back( std::get<int>(v) );
      }
    }

    while(batch.batch_size < batch_size) {

      // pick random number of bars from domain
      if (enc->rep->has_token_type(NUM_BARS)) {
        int index = random_on_range(num_bar_choices.size(), &engine);
        tc->num_bars = num_bar_choices[index];
      }

      try {
        midi::Piece p;
        load_random_segment(&p, split_id, enc.get(), tc);
        std::vector<int> tokens = enc->encode(&p);
        std::vector<int> mask(tokens.size(),1);
        batch.add( tokens );
        att_mask.add( mask );
      }
      catch (const std::exception &exc)
      {
        std::cerr << exc.what() << std::endl;
      }
    }
    batch.pad(0);
    att_mask.pad(0);
    return make_tuple(batch.batch, att_mask.batch);
  }

  std::tuple<matrix<int>,matrix<int>,tensor<double>> read_batch_w_feature(int batch_size, size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {
    enable_read();
    std::unique_ptr<ENCODER> enc = getEncoder(et);
    if (!enc) {
      throw std::runtime_error("ENCODER TYPE DOES NOT EXIST");
    }

    Batcher<int> batch(max_seq_len, &engine);
    Batcher<int> att_mask(max_seq_len, &engine);
    Batcher<std::vector<double>> feature(max_seq_len, &engine);

    while(batch.batch_size < batch_size) {
      try {
        midi::Piece p;
        load_random_segment(&p, split_id, enc.get(), tc);
        auto out = enc->encode_w_embeds(&p);
        std::vector<int> mask(std::get<0>(out).size(),1);
        batch.add( std::get<0>(out) );
        att_mask.add( mask );
        feature.add( std::get<1>(out) );
      }
      catch (const std::exception &exc)
      {
        std::cerr << exc.what() << std::endl;
      }
    }
    batch.pad(0);
    att_mask.pad(0);
    feature.pad( enc->empty_embedding() );
    return make_tuple(batch.batch, att_mask.batch, feature.batch);
  }

  std::tuple<std::vector<std::vector<int>>,std::vector<std::vector<int>>> read_batch(int batch_size, size_t split_id, ENCODER_TYPE et, TrainConfig *tc) {
    enable_read();
    midi::Piece x;
    int index;
    int nitems = get_split_size(split_id);
    int maxlen = 0;

    std::unique_ptr<ENCODER> enc = getEncoder(et);
    if (!enc) {
      throw std::runtime_error("ENCODER TYPE DOES NOT EXIST");
    }

    std::vector<std::vector<int>> mask;
    std::vector<std::vector<int>> batch;

    while (batch.size() < batch_size) {

      try {

        index = random_on_range(nitems, &engine);
        std::string serialized_data = read(index, split_id);
        x.ParseFromString(serialized_data);

        // pick random segment
        select_random_segment(&x, tc->num_bars, tc->min_tracks, tc->max_tracks, enc->config->te, &engine);

        // pick random transpose
        std::tuple<int,int> pitch_ext = get_pitch_extents(&x);
        //cout << std::get<0>(pitch_ext) << " -- " << std::get<1>(pitch_ext) << std::endl;
        std::vector<int> choices;
        for (int tr=-6; tr<6; tr++) {
          if ((std::get<0>(pitch_ext)+tr >= 0) && (std::get<1>(pitch_ext)+tr < 128)) {
            choices.push_back( tr );
          }
        }
        //enc->config->transpose = choices[rand() % choices.size()];
        enc->config->transpose = choices[random_on_range(choices.size(),&engine)];

        // 75 % of the time we do bar infill
        // the old code actually only 25 %
        if (enc->config->both_in_one) {
          //enc->config->do_multi_fill = (rand() % 1000 > 750);
          enc->config->do_multi_fill = random_on_unit(&engine) < .75;
        }

        // pick bars for infilling
        if (enc->config->do_multi_fill) {
          enc->config->multi_fill = make_bar_mask(
            &x, tc->max_mask_percentage, &engine);
        }

        // encode tokens
        std::vector<int> raw_tokens = enc->encode(&x);

        // reject partially full sequences
        /*
        if (et == TRACK_SEGMENT_ENCODER) {
          if (raw_tokens.size() < max_seq_len) {
            throw std::runtime_error("TOKEN SEQ NOT LONG ENOUGH");
          }
          raw_tokens.resize(max_seq_len);
        }
        */

        std::vector<int> tokens;
        if ((raw_tokens.size() > max_seq_len) && (!tc->no_max_length)) {
          // pick a random section
          //int offset = rand() % (raw_tokens.size() - max_seq_len + 1);
          int offset = random_on_range(
            (int)raw_tokens.size() - max_seq_len + 1, &engine);
          copy(
            raw_tokens.begin() + offset, 
            raw_tokens.begin() + offset + max_seq_len,
            back_inserter(tokens));
        }
        else {
          copy(raw_tokens.begin(), raw_tokens.end(), back_inserter(tokens));
        }
        batch.push_back( tokens );
        mask.push_back( std::vector<int>(tokens.size(),1) );
        maxlen = std::max((int)tokens.size(), maxlen);
      }
      catch (const std::exception &exc)
      {
          //std::cerr << exc.what() << std::endl;
      }

    }
    // right pad the sequences
    for (int i=0; i<batch_size; i++) {
      batch[i].insert(batch[i].end(), maxlen-batch[i].size(), 0);
      mask[i].insert(mask[i].end(), maxlen-mask[i].size(), 0);
      assert(mask[i].size() == maxlen);
      assert(batch[i].size() == maxlen);
    }
    return std::make_pair(batch,mask);
  }

  int get_size() {
    enable_read();
    return header.train_size() + header.valid_size() + header.test_size();
  }

  int get_split_size(int split_id) {
    enable_read();
    switch (split_id) {
      case 0 : return header.train_size();
      case 1 : return header.valid_size();
      case 2 : return header.test_size();
    }
    return 0; // invalid split id
  }

  void flush() {
    fs.flush();
    header_fs.open(header_filepath, std::ios::out | std::ios::binary);
    if (!header.SerializeToOstream(&header_fs)) {
      std::cerr << "ERROR : Failed to write header file" << std::endl;
    }
    header_fs.close();
  }

  void close() {
    flush();
    fs.close();
    header_fs.close();
    can_read = false;
    can_write = false;
  }
  
private:
  std::string filepath;
  std::string header_filepath;
  std::fstream fs;
  std::fstream header_fs;
  bool can_write;
  bool can_read;
  midi::Dataset header;
  int flush_count;

  int num_bars;
  int min_tracks;
  int max_tracks;
  int max_seq_len;

  std::mt19937 engine;

  std::vector<std::vector<int>> bstore;
  ENCODER *encoder;
};

}
// END OF NAMESPACE



