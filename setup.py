from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='xfc_gemm',
    ext_modules=[
        CUDAExtension('xfc_gemm_cuda', [
            'xfc_gemm.cpp',
            'xfc_gemm_kernel.cu',
        ],
#         include_dirs=['./include','./tools/util/include'],
        include_dirs=[f'{os.getcwd()}/cutlass/include',f'{os.getcwd()}/cutlass/tools/util/include'],
        extra_compile_args={'cxx': ['-O3'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
