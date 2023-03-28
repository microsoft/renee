#include <torch/extension.h>

void xfc_gemm_cuda(
    torch::Tensor mat_in1,
    torch::Tensor mat_in2,
    torch::Tensor mat_out,
    float alpha,
    float beta,
    bool apply_sigmoid);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void xfc_gemm(
    torch::Tensor mat_in1,
    torch::Tensor mat_in2,
    torch::Tensor mat_out,
    float alpha,
    float beta,
    bool apply_sigmoid){
  //CHECK_INPUT(mat_in1); 
  //CHECK_INPUT(mat_in2); 
  //CHECK_INPUT(mat_out); 
  
  AT_ASSERTM(mat_in1.dim()   == 2, "expected 2D tensor");
  AT_ASSERTM(mat_in2.dim()   == 2, "expected 2D tensor");
  AT_ASSERTM(mat_out.dim()   == 2, "expected 2D tensor");

  //AT_ASSERTM(mat_in1.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  //AT_ASSERTM(mat_in2.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
   
  xfc_gemm_cuda(mat_in1, mat_in2, mat_out, alpha, beta, apply_sigmoid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("xfc_gemm", &xfc_gemm, "Optimized gemm.");
}

