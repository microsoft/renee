
#run below cmd first
#conda activate py38

pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia


pip install transformers
pip install scipy
pip install threadpoolctl==2.0.0
pip install pandas
pip install cython

pip install sentence-transformer
pip install pybind11==2.6.1 fasttext cython numpy==1.20.3 scikit-learn


#git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
python3 setup.py install  
cd ..

#git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..


pip install numba

# git clone -b python_bindings_quantized https://github.com/microsoft/DiskANN.git
# Make sure that MKL is installed: see repo instructions
# cd DiskANN
# sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev clang-format-4.0 libboost-dev
# mkdir build && cd build && cmake .. && make -j 
# cd ../python
# pip install -e .

