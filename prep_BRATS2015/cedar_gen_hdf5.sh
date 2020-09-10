#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # number of MPI processes
#SBATCH --cpus-per-task=2      # 24 cores on cedar nodes
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=16G                 # give all memory you have in the node
#SBATCH --time=3-05:00         # time (DD-HH:MM)
#SBATCH --job-name=GenerateHDF5File
#SBATCH --output=GenerateHDF5File.out
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL

# run the command
~/.virtualenvs/mm_synthesis/bin/python create_hdf5_file.py
