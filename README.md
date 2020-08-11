Overview
--------
This is a repository for FlexZBoost in DESC Pipeline compatible stage

There are two components to the process:
- There is a script named `train_FlexZBoost.py` that takes in a spec-z training file and outputs a pickled model.  When the model/pickle file is loaded, most of the parameter settings can be viewed with `[modelname].__dict__`

- There is the actual class FlexZPipe that takes the trained model from train_FlexZBoost.py and runs the pipeline stage.  This will be designed to be very similar to BPZPipe for easy comparison.

Model
-----

A pre-trained model can be downloaded from NERSC:

```
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/flexcode_model.pkl
```
