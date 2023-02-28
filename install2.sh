
#run below cmd first before exec'ing this file
#conda activate py38

conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia


pip install transformers
pip install scipy
pip install threadpoolctl==2.0.0
pip install pandas
pip install cython
pip install pyarrow
pip install sklearn
pip install zipp
#conda install -c numba llvmlite


#git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
python3 setup.py install  
cd ..

#git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
