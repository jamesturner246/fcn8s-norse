# load the corresponding modules if not already loaded
module load daint-gpu
module load singularity # or singularity/3.6.4-daint

# Download data
wget https://kth.box.com/shared/static/7upvu4qmg12nu61q7d0hhf16lq34uccj.dat -O scenes_60.dat

# Train model
python norse-dvs.py --gpus=1
