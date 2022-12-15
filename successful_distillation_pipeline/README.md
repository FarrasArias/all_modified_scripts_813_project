# Code Modifications

## Edits
In script build_dataset.py the encoder was changed on line 19 to TrackDensityEncoder

In script jagged.h line 71 max_seq_len was changed to 1024.

In script train.py there are several modifications
- line 121: The DistilGPT2 pretrained model is loaded for training
- line 138-142: These lines load a checkpoint for the testing phase (inference times testing)
- line 184-203: These lines perform the time tests for random inferences for the testing phase
- 206: An evaluation line was added in training to look for inference times during training
