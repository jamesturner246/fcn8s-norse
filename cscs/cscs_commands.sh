# load the corresponding modules if not already loaded
module load daint-gpu
module load singularity # or singularity/3.6.4-daint

# Train model
srun -C gpu --account=course00 singularity pull docker://jegp/cscs:latest
srun -C gpu --account=course00 singularity exec jegp_cscs_latest
