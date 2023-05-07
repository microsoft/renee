# run below cmd first
# conda activate renee

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install transformers
pip install scipy
pip install numba
pip install pandas
pip install cython
pip install scikit-learn
pip install sentence-transformers
pip install seaborn

git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
pip install .
cd ..
# rm -rf pyxclib

## Need only for apex optimizers
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
# rm -rf apex


## Need only for custom-cuda
# git clone https://github.com/NVIDIA/cutlass.git
# pip install cutlass
# python3 setup.py install
