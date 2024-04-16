#!/bin/bash
SBATCH --job-name=test_job 	# Job name
SBATCH --partition=test 		#Partition name can be test/small/medium/large/gpu/gpu2 #Partition “gpu or gpu2” should be used only for gpu jobs
SBATCH --nodes=1 				# Run all processes on a single node
SBATCH --ntasks=1 				# Run a single task
SBATCH --cpus-per-task=4 		# Number of CPU cores per task
SBATCH --gres=gpu 				# Include gpu for the task (only for GPU jobs)
SBATCH --mem=6gb 				# Total memory limit (optional)
SBATCH --time=00:10:00 		# Time limit hrs:min:sec (optional)
SBATCH --output=first_%j.log 	# Standard output and error log
date;hostname;pwd


module load anaconda/3

python model.py