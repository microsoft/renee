
#run below cmd first
#conda activate rpr

pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html


pip install transformers
pip install scipy
pip install threadpoolctl==2.0.0
pip install pandas
pip install cython

pip install seaborn sentence-transformer
pip install --no-binary :all: nmslib
pip install pybind11==2.6.1 fasttext sklearn cython numpy==1.20.3 scikit-learn


#git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
python3 setup.py install  
cd ..

#git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..


pip install numba

