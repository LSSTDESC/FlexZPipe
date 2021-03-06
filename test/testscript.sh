#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:29:59
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --error="testfzboost23newpath.err"
#SBATCH --output="testfzboost23newpath.out"

module load python/3.7-anaconda-2019.07
module swap PrgEnv-intel PrgEnv-gnu
module load PrgEnv-gnu
module unload darshan
module load h5py-parallel
module load cfitsio/3.47
module load gsl/2.5


export CECI_SETUP="/global/projecta/projectdirs/lsst/groups/PZ/FlexZBoost/FlexZPipe/setup-flexz-cori-update"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/global/projecta/projectdirs/lsst/groups/PZ/FlexZBoost/install/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/FlexZBoost/FlexZPipe:/global/projecta/projectdirs/lsst/groups/PZ/Packages/descformats/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/ceci/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/parsl0.5.2/lib/python3.7/site-packages


srun -n 8 python3 -m flexzpipe FlexZPipe --photometry_catalog=./z_2_3.step_all.healpix_10326_magwerrSNtrim.hdf5 --config=./config.yml --photoz_pdfs=./outputs/z_2_3_testfile_pdfs_newpath.hdf5 --mpi
