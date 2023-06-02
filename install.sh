# install miniconda3 if not installed yet.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# switch to CUDA toolkit version 11.3
source switch-cuda.sh 11.3

# install PyTorch
conda deactivate
conda env remove --name openpoints
conda create -n openpoints -y python=3.7 numpy=1.20 numba
conda activate openpoints

conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

# install relevant packages
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../../..

