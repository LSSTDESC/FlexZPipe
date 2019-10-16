import sys
import numpy as np
import h5py
import yaml
import pandas as pd
import flexcode
from flexcode.regression_models import XGBoost
import pickle
from flexcode.loss_functions import cde_loss

def read_in_data(trainfile):
    """
    read in data from file, will assume that input data has same format 
    expected for BPZPipe: h5py hdf5 file with ugrizy mags and errors plus
    redshift.
    input:
      trainfile: name of training hdf5 file (ascii string)
    output:
      datavec: numpy nd-array of photometry inputs to flexcode model
      szarray: numpy 1d-array of redshifts for flexcode model training
    """
    filts = ['u','g','r','i','z','y']
    numfilts = len(filts)
    f = h5py.File(trainfile,"r")
    p = f['photometry']
    sz = np.array(p['redshift'])
    i = p['mag_i_lsst'] #grab the i-band magnitude
    tmpdict = {'i':i}
    df = pd.DataFrame(tmpdict)
    for xx in range(numfilts-1):
        df[f'{filts[xx]}'f'{filts[xx+1]}']= \
        np.array(p[f'mag_{filts[xx]}_lsst']) -\
        np.array(p[f'mag_{filts[xx+1]}_lsst'])

        df[f'{filts[xx]}'f'{filts[xx+1]}_err'] = np.sqrt(\
        np.array(p[f'mag_err_{filts[xx]}_lsst'])**2.0 +\
        np.array(p[f'mag_err_{filts[xx+1]}_lsst'])**2.0)

    return df.to_numpy(), sz

def partition_data(fz_data,sz_data,trainfrac):
    """
    make a random partition of the training data into training and validation
    """
    nobs = fz_data.shape[0]
    ntrain = round(nobs*trainfrac)
    nvalidate = nobs - ntrain
    perm = np.random.permutation(nobs)
    x_train = fz_data[perm[:ntrain],:]
    z_train = sz_data[perm[:ntrain]]
    x_val = fz_data[perm[ntrain:]]
    z_val = sz_data[perm[ntrain:]]
    
    return x_train,x_val,z_train,z_val
    
def main(argv):
    if len(argv) != 2:
        print("usage: train_FlexZBoost.py [yamlfile]")
        exit()
    else:
        infile = argv[1]
    with open(infile,"r") as infp:
        ymldata = yaml.load(infp)
    
    
    output_file = ymldata['output_file']
    validationfile = ymldata['sharpen_bumpthresh_outputfile']
    #trainfile = "z_2_3.step_all.healpix_10447_magwerrSNtrim.hdf5"
    trainfile = ymldata['training_file']
    max_basis = ymldata['max_basis_functions']
    basis_system = ymldata['basis_system']
    z_min = float(ymldata['z_min'])
    z_max = float(ymldata['z_max'])
    regression_params = ymldata['regression_params']
    bumpmin = float(ymldata['bump_thresh_grid_min'])
    bumpmax = float(ymldata['bump_thresh_grid_max'])
    bumpdelta = float(ymldata['bump_thresh_grid_delta'])
    train_frac = float(ymldata['training_fraction'])
    sharpmin = float(ymldata['sharpen_min'])
    sharpmax = float(ymldata['sharpen_max'])
    sharpdelta = float(ymldata['sharpen_delta'])
    
    bump_grid = np.arange(bumpmin,bumpmax,bumpdelta)
    print("read in training data")
    fz_data, sz_data = read_in_data(trainfile)
    print("partition into train and validate")
    fz_train,fz_val,sz_train,sz_val = partition_data(fz_data,sz_data,train_frac)
    print (fz_train.shape[0])
    print("train the model")
    model = flexcode.FlexCodeModel(XGBoost,max_basis=max_basis,
                                   basis_system=basis_system, z_min=z_min,
                                   z_max=z_max,
                                   regression_params=regression_params)
    model.fit(fz_train,sz_train)
    print("tune model, including bump trimming")
    #running with a grid as input wasn't working as it should, just add loop
    #model.tune(fz_val,sz_val,bump_threshold_grid=bump_grid)
    #the tuning computes the CDE loss for each bump_threshold in the grid and
    #chooses the best value based on the validation data

    #NOTE: sample runs on two samples both chose lowest possible bump thresh
    #do a brute force loop and spit out the CDE loss to make sure that the
    #lowest bump thresh really has the best loss score
    outfp = open(validationfile,"w")
    outfp.write("CDE Loss values for bump thresh and sharpen grids\n")
    bestloss = 9999
    for bumpt in bump_grid:
        model.bump_threshold=bumpt
        model.tune(fz_val,sz_val)
        tmpcdes,z_grid = model.predict(fz_val,n_grid=300)
        tmploss = cde_loss(tmpcdes,z_grid,sz_val)
        if tmploss < bestloss:
            bestloss = tmploss
            bestbump = bumpt
        print(f"\n\n\nbumptrim val: {bumpt} cde loss: {tmploss}")
        outfp.write(f"bumptrim val: {bumpt} cde loss: {tmploss}\n")
    print(f"\n\n\nbest bump threshold: {bestbump} setting in model\n\n\n")
    model.bump_threshold=bestbump
    #now do the same for sharpening parameter!
    #    sharpen_grid = np.arange(0.8,2.101,0.1)
    sharpen_grid = np.arange(sharpmin,sharpmax,sharpdelta)
    bestloss = 9999
    bestsharp = 9999
    for sharp in sharpen_grid:
        model.sharpen_alpha = sharp
        tmpcdes,z_grid = model.predict(fz_val,n_grid=301)
        tmploss = cde_loss(tmpcdes,z_grid,sz_val)
        if tmploss < bestloss:
            bestloss = tmploss
            bestsharp = sharp
        print(f"\n\n\nsharpparam: {sharp} cdeloss: {tmploss}")
        outfp.write(f"sharpparam: {sharp} cdeloss: {tmploss}\n")
    print(f"best sharpen param: {bestsharp}")
    model.sharpen_alpha=bestsharp
    # Saving the model
    pickle.dump(file=open(output_file, 'wb'), obj=model, 
                protocol=pickle.HIGHEST_PROTOCOL)
    print (f"wrote out model file file to {output_file}")
    outfp.close()
    print(model.__dict__)

    print("finished")
if __name__=="__main__":
    main(sys.argv)
