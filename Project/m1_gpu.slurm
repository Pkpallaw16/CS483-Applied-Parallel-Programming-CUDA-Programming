#!/bin/bash
#SBATCH --job-name="Milestone1_GPU"
#SBATCH --output="Milestone1_GPU.out"
#SBATCH --error="Milestone1_GPU.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects" #,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

#srun ./m1_gpu 10000 > m1_gpu10000.out

# Generate nsys profile information, output will be in profile.out
#srun nsys profile --stats=true ./m1_gpu > profile10000.out
#srun ncu -f -o analysis_file ./m1_gpu 10000
#srun ncu --set full -f -o analysis_file ./m1_gpu 10000
#{time srun ./m1_gpu 100 >m1_gpu_100.out;} 2>gpu-time.out
{ time srun ./m1_gpu 10000 > m1_gpu_tenthou.out; } 2> time_tenthou.out	
