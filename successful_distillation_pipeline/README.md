# Code Modifications

## Edits
In script build_dataset.py the encoder was changed on line 19 to TrackDensityEncoder

In script jagged.h line 71 max_seq_len was changed to 1024.

In script train.py there are several modifications
- line 121: The DistilGPT2 pretrained model is loaded for training
- line 138-142: These lines load a checkpoint for the testing phase (inference times testing)
- line 184-203: These lines perform the time tests for random inferences for the testing phase
- 206: An evaluation line was added in training to look for inference times during training

Script run_train_or_test.sh is made to configure Compute Canada to run an sbatch job and train or test the model.

In order to run the code, these scripts must replace their original scripts.

jagged.h must replace jagged.h in https://github.com/jeffreyjohnens/MMM_API/tree/master/src/mmm_api/dataset
build_dataset.py must replace build dataset.py in https://gitlab.com/jeffreyjohnens/MMM_TRAINING/-/blob/master/build_dataset.py
train.py must replace train.py in https://gitlab.com/jeffreyjohnens/MMM_TRAINING/-/blob/master/train.py
run_train_or_test.sh must be added to https://gitlab.com/jeffreyjohnens/MMM_TRAINING/-/tree/master/

To compile the MMM_API and train a model, instructions are in the following repo: https://github.com/FarrasArias/temp_CC_Metalab
**NOTE:** This last repo was not created for the 813 class. The instructions where developed by me (Rafael Arias) and I have maintained and updated them, but they were created before for other uses of the MMM codebase.
