#include <vector>
#include <iostream>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/device_memory.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


void xfc_gemm_cuda(
    torch::Tensor mat_in1,
    torch::Tensor mat_in2,
    torch::Tensor mat_out, 
    float alpha_in,
    float beta_in,
    bool apply_sigmoid) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float; // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??
// Number of pipelines you want to use
constexpr int NumStages = 2;
ElementComputeEpilogue alpha = ElementComputeEpilogue(alpha_in);
ElementComputeEpilogue beta = ElementComputeEpilogue(beta_in);

int M = mat_out.size(0);
int N = mat_out.size(1);
int K = mat_in2.size(0);
/*
std::cout << "matin1: " << mat_in1.size(0)  << ", " <<mat_in1.size(1) << ", " << mat_in1.stride(0) << std::endl;
std::cout << "matin2: " << mat_in2.size(0)  << ", " <<mat_in2.size(1) << ", " << mat_in2.stride(0) << std::endl;
std::cout << "matOut: " << mat_out.size(0)  << ", " <<mat_out.size(1) << ", " << mat_out.stride(0) << std::endl;
std::cout << "M,N,K: " << mat_out.size(0)  << ", " <<mat_out.size(1) << ", " <<mat_in2.size(0) << std::endl;
*/

 if (apply_sigmoid)  {
  int split_k_slices = 1;
  using ElementOutputH = cutlass::half_t;    // <- data type of elements in output matrix D
  using EpilogueOpSH = cutlass::epilogue::thread::LinearCombinationSigmoid<
    ElementOutputH, 128 / cutlass::sizeof_bits<ElementOutputH>::value, ElementAccumulator,
    ElementComputeEpilogue>;  
  if ((mat_in1.stride(1) == 1) && (mat_in2.stride(0) == 1)) {
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile, e.g. M = 128, N = 128, K = 32
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile,e.b. M = 64, N = 64, K = 32
    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutputH,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOpSH,
                                         SwizzleThreadBlock,
                                         NumStages>;

    Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
	    {static_cast<const cutlass::half_t *>(mat_in1.data_ptr()),mat_in1.stride(0)},  // <- reference to matrix A on device
	    {static_cast<const cutlass::half_t *>(mat_in2.data_ptr()),mat_in2.stride(1)},  // <- reference to matrix B on device
	    {static_cast<cutlass::half_t *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to matrix C on device
	    {static_cast<cutlass::half_t *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to matrix C on device
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices);        // <- k-dimension split factor
    
    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    CUTLASS_CHECK(status);
   }
   else {
	   std::cout << "Only rowXcolumn supported for applysigmoid=True. TBD" << std::endl;
	   exit(-1);
   }

 }  // end if apply_sigmoid
 else { 
  using ElementOutputF = float;    // <- data type of elements in output matrix D
  using EpilogueOpDF = cutlass::epilogue::thread::LinearCombination<
    ElementOutputF, 128 / cutlass::sizeof_bits<ElementOutputF>::value, ElementAccumulator,
    ElementComputeEpilogue>;  

  if ((mat_in1.stride(0) == 1) && (mat_in2.stride(1) == 1)) { 
    // Gradient with respect to weights
    int split_k_slices = 1;
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutputF,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOpDF,
                                         SwizzleThreadBlock,
                                         NumStages>;

    Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
	    {static_cast<const cutlass::half_t *>(mat_in1.data_ptr()),mat_in1.stride(1)},  // <- reference to matrix A on device
	    {static_cast<const cutlass::half_t *>(mat_in2.data_ptr()),mat_in2.stride(0)},  // <- reference to matrix B on device
	    {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to matrix C on device
	    {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to matrix C on device
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices);        // <- k-dimension split factor
    

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    CUTLASS_CHECK(status);
  }
  else if ((mat_in1.stride(1) == 1) && (mat_in2.stride(1) == 1)) {
    //Gradient with respect to input 
    /*
    int split_k_slices = 1;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutputF,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOpDF,
                                         SwizzleThreadBlock,
                                         NumStages>;

    Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
            {static_cast<const cutlass::half_t *>(mat_in1.data_ptr()),mat_in1.stride(0)},  // <- reference to trans of matrix B on device
            {static_cast<const cutlass::half_t *>(mat_in2.data_ptr()),mat_in2.stride(0)},  // <- reference to trans of matrix A on device
            {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to trans of matrix C on device
            {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to trans of matrix C on device
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices);        // <- k-dimension split factor


    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    CUTLASS_CHECK(status);
    */ 
     
    // Gradient with respect to inputs, K is large
    int split_k_slices = 128;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutputF,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOpDF>;
                                         //SwizzleThreadBlock,
                                         //NumStages>;


    Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
	    {static_cast<const cutlass::half_t *>(mat_in1.data_ptr()),mat_in1.stride(0)},  // <- reference to trans of matrix B on device
	    {static_cast<const cutlass::half_t *>(mat_in2.data_ptr()),mat_in2.stride(0)},  // <- reference to trans of matrix A on device
	    {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to trans of matrix C on device
	    {static_cast<float *>(mat_out.data_ptr()),mat_out.stride(0)}, // <- reference to trans of matrix C on device
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices);        // <- k-dimension split factor
    
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(args);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Initialize CUTLASS kernel with arguments and workspace pointer
    cutlass::Status status = gemm_op.initialize(args, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);
    
    
  }
  else {
	   std::cout << "Only columnxrow and rowxrow  supported for applysigmoid=False. TBD" << std::endl;
	   exit(-1);
  }
 }  // end else apply_sigmoid
}
