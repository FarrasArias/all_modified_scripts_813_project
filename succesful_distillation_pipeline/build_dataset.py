import os
import glob
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from utils import *
import mmm_api as mmm

def worker(args):
	path,sid,genre,tcjson = args
	tc = mmm.TrainConfig()
	tc.from_json(tcjson)
	if tc.opz:
		encoder = mmm.TeTrackDensityEncoder()
	else:
		#encoder = mmm.TrackEncoder()
                encoder = mmm.TrackDensityEncoder()
	try:
		return sid, encoder.midi_to_json_bytes(path,tc,json.dumps(genre))
	except Exception as e:
		print(e)
		return None,None

BLANK_GENRE_PROTOBUF = {"tagtraum":"NONE", "discogs":"NONE", "lastfm":"NONE"}
def get_supergenre(genres):
	if len(genres):
		counts = {}
		for genre,count in genres.items():
			supergenre = genre.split("---")[0]
			counts[supergenre] = counts.get(supergenre,0) + count
		g = np.array(list(counts.keys()))
		c = np.array(list(counts.values()))
		if np.sum(c==np.max(c))==1:
			return g[np.argmax(c)]
	return "NONE"

def load_genre(genre_path):
	genre_map = {}
	for item in load_jsonl(genre_path):
		genre_protobuf = {
			"tagtraum" : get_supergenre(item.get("genre_tagtraum",[])),
			"lastfm" : get_supergenre(item.get("genre_lastfm",[])),
			"discogs" : get_supergenre(item.get("genre_discogs",[]))
		}
		genre_map[item["md5"]] = genre_protobuf
	return genre_map

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--num_bars", type=int, default=4)
	parser.add_argument("--opz", type=int, default=0)
	parser.add_argument("--nthreads", type=int, default=8)
	parser.add_argument("--max_size", type=int, default=-1)
	parser.add_argument("--genre_data", type=str, default="")
	args = parser.parse_args()

	args.opz = bool(args.opz)

	import os
	os.system("taskset -p 0xffff %d" % os.getpid())

	# multi thread approach takes about 2 minutes
	pool = Pool(args.nthreads)
	output = os.path.splitext(args.output)[0]
	output += "_NUM_BARS={}_OPZ_{}.arr".format(args.num_bars,args.opz)
	jag = mmm.Jagged(output)
	paths = list(glob.glob(args.data_dir + "/**/*.mid", recursive=True))
	
	import random
	import time
	random.seed(int(time.time()))

	tc = mmm.TrainConfig()
	tc.num_bars = args.num_bars
	tc.opz = args.opz
	tc = tc.to_json()
	
	# when dataset is split
	sids = []
	for path in paths:
		if "/train/" in path:
			sids.append(0)
		elif "/valid/" in path:
			sids.append(1)
		elif "/test/" in path:
			sids.append(2)
		else:
			raise RuntimeError("data format incorrect")

	genre_map = {}
	if args.genre_data:
		genre_map = load_genre(args.genre_data)

	genres = [genre_map.get(os.path.splitext(os.path.basename(p))[0],BLANK_GENRE_PROTOBUF) for p in paths]
	tcs = [tc for _ in paths]
	inputs = list(zip(paths,sids,genres,tcs))
	random.shuffle(inputs)

	if args.max_size > 0:
		inputs = inputs[:args.max_size]

	total_count = 0
	success_count = 0
	pool = Pool(args.nthreads)
	progress_bar = tqdm(pool.imap_unordered(worker, inputs), total=len(paths))
	for sid,b in progress_bar:
		if b is not None and len(b):
			jag.append(b,sid)
			success_count += 1
		total_count += 1
		status_str = "{}/{}".format(success_count,total_count)
		progress_bar.set_description(status_str)
	jag.close()
