from transformers import *

import os
import json
import time
import torch

import datetime
import numpy as np
from tqdm import tqdm

import mmm_api as mmm
from losses import sim_metric_loss, standard_loss
from custom_models import *
from train_dataset import *
from transformers import Trainer, TrainingArguments
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

# TODO handle case with NAN in loss

if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--arch", type=str, required=True)
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--encoding", type=str, required=True)
  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument("--pad_value", type=int, default=-100)

  parser.add_argument("--opz", action="store_true")
  parser.add_argument("--num_bars", type=int, default=4)
  parser.add_argument("--min_tracks", type=int, default=2)
  parser.add_argument("--max_tracks", type=int, default=12)
  parser.add_argument("--max_seq_len", type=int, default=1024)
  parser.add_argument("--no_max_length", type=int, default=0)

  parser.add_argument("--ngpu", type=int, default=4)
  parser.add_argument("--accum_steps", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--batches_per_epoch", type=int, default=1000)
  parser.add_argument("--lr", type=float, default=1e-4)

  parser.add_argument("--overwrite", type=int, default=1)
  parser.add_argument("--save_steps", type=int, default=5000)
  parser.add_argument("--log_steps", type=int, default=100)
  parser.add_argument("--step", type=int, default=0)
  parser.add_argument("--label", type=str, default="version3")

  parser.add_argument("--dry", action="store_true")
  parser.add_argument("--metric", action="store_true")

  parser.add_argument("--ckpt", type=str, default="")

  args = parser.parse_args()

  dataset_cls = CustomDataset
  loss_fn = standard_loss
  if args.arch == "metric":
    args.no_max_length = 1
    dataset_cls = EncoderDataset
    loss_fn = sim_metric_loss
  if args.arch == "control":
    dataset_cls = FeatureDataset
  
  np.random.seed(int(time.time()))

  # determine vocab size
  date_str = datetime.datetime.now().strftime('%b_%d_%H_%M')
  encoder_mode = mmm.getEncoderType(args.encoding)
  assert encoder_mode is not mmm.ENCODER_TYPE.NO_ENCODER
  encoder = mmm.getEncoder(encoder_mode)
  vocab_size = encoder.rep.vocab_size
  print("VOCAB",vocab_size)
  name = "_".join([args.encoding, args.arch, args.label, date_str, "num_bars", str(args.num_bars), str(args.max_tracks)])

  if args.dry:
    while True:
      dataset = dataset_cls(split_id=0, is_training=True, **vars(args))
      for batch in tqdm(dataset,smoothing=0):
        np_inputs = batch["input_ids"].detach().numpy()
        print( [encoder.rep.pretty(t) for t in np_inputs[0][:100]] )
        print( {k:v.shape for k,v in batch.items()} )
  
  if os.getenv("SLURM_TMPDIR") is not None:
    # we are on compute canada and should attempt to copy 
    # dataset to tmpdir for faster access
    from shutil import copyfile
    tmpdir = os.getenv("SLURM_TMPDIR")
    dataset_path = os.path.join(tmpdir, os.path.basename(args.dataset))
    if not os.path.exists(dataset_path):
      copyfile(args.dataset, dataset_path)
      copyfile(args.dataset + ".header", dataset_path + ".header")
    args.dataset = dataset_path

  # setup datasets
  train_dataset = dataset_cls(split_id=0, is_training=True, **vars(args))
  eval_dataset = dataset_cls(split_id=1, is_training=False, overload_batches_per_epoch=1, **vars(args))
  Trainer.get_train_dataloader = lambda *_args,**_kwargs: train_dataset
  Trainer.get_eval_dataloader = lambda  *_args,**_kwargs: eval_dataset
  Trainer.compute_loss = loss_fn

  

  print("MODEL NAME : " + name)
  print("VOCAB SIZE : " + str(vocab_size))
  print("ARGS : " + json.dumps(vars(args),indent=4))
  print("CONFIG : " + json.dumps(json.load(open(args.config,"r")),indent=4))

  logging_dir = "logs/{}".format(name)
  output_dir = "checkpoints/{}".format(name)
  
  os.makedirs(logging_dir, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)

  # =================================================================
  # model selection

  if args.arch == "gpt2":
    config = GPT2Config().from_json_file(args.config)
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.max_seq_length = 1024
  elif args.arch == "xl":
    config = TransfoXLConfig().from_json_file(args.config)
    model_cls = TransfoXLLMHeadModel
  elif args.arch == "metric":
    config = GPT2Config().from_json_file(args.config)
    model_cls = GPT2Encoder
  elif args.arch == "control":
    config = GPT2LMHeadModelContConfig().from_json_file(args.config)
    # encoder knows the size of the embedding
    config.n_control_dim = encoder.config.embed_dim 
    model_cls = GPT2LMHeadModelCont
  else:
    raise NotImplementedError 
  
  config.vocab_size = vocab_size
  ####
  # Uncomment the collowing line and add a checkpoint for tests
  ####
  #args.ckpt = "/EL_VELOCITY_DURATION_POLYPHONY_ENCODER_gpt2_GENRE_DISCOGS_Dec_10_16_04_num_bars_4_12/checkpoint-6912"

  if len(args.ckpt.strip()) == 0:
    ckpt_path = None
    model = model_cls(config)
  else:
    #ckpt_path = os.path.join(output_dir, "checkpoint-{}".format(args.step))
    ckpt_path = os.path.join("checkpoints/"+args.ckpt)
    model = model.from_pretrained(ckpt_path)

  # =================================================================
  # training 
  
  training_args = TrainingArguments(
    logging_dir=logging_dir,
    report_to="tensorboard",
    output_dir=output_dir,
    overwrite_output_dir=bool(args.overwrite),
    num_train_epochs=(500000/args.batches_per_epoch)*args.accum_steps,
    #num_train_epochs=5,
    logging_steps=args.log_steps,
    #save_steps=args.save_steps,
    save_steps=1,
    save_total_limit=None,
    learning_rate=args.lr,
    gradient_accumulation_steps=args.accum_steps,
    per_device_train_batch_size=args.batch_size//args.ngpu//args.accum_steps,
    per_device_eval_batch_size=args.batch_size//args.ngpu//args.accum_steps,
    evaluation_strategy="epoch",
    prediction_loss_only=True,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
  )

  trainer.train_dataset = train_dataset
  trainer.eval_dataset = eval_dataset

  ############
  #This code is for the Inference Time Tests
  ####
  #for inputs in trainer.eval_dataset:
  #  ins = inputs
  #  print(inputs)
  #  print(len(inputs["input_ids"]))
  #  break

  #times = []
  #for counter in range(32):
  #  insi = {"input_ids":ins["input_ids"][counter],"attention_mask":ins["attention_mask"][counter], "labels":ins["labels"][counter]}
  #  print(len(insi["input_ids"]))
  #  t = time.time()
  #  trainer.predict([insi])
  #  times.append(time.time()-t)
  #print(times)
  ####
  #End of inference time tests
  ############
  
  trainer.train(ckpt_path)
  trainer.evaluate()
  
