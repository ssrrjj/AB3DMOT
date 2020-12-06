source /team1/envs/conda/bin/activate
export PYTHONPATH=${PYTHONPATH}:/team1/codes/AB3DMOT_renjie
export PYTHONPATH=${PYTHONPATH}:/team1/codes/Xinshuo_PyToolbox/
pip install -r waymo_requirements.txt
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev