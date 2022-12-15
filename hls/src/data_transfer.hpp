
// data_transfer.hpp

#ifndef DATA_TRANSFER_HPP
#define DATA_TRANSFER_HPP

#include "data_conversion.hpp"
#include "layer_types.hpp"

#include <cassert>

// Read a point from a DDR memory
template <typename T>
void ReadPointNaive(const float* point_cloud,
                    const int idx,
                    T x[3])
{
#pragma HLS INLINE off

  for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE II=1
    x[i] = T(point_cloud[idx * 3 + i]);
  }
}

// Read a point from a DDR memory
template <typename T>
void ReadPointOpt1(const ap_uint<64>* point_cloud,
                   const int idx,
                   T x[3])
{
#pragma HLS INLINE off

  const ap_uint<64> point_data0 = point_cloud[idx * 2 + 0];
  const ap_uint<64> point_data1 = point_cloud[idx * 2 + 1];
  x[0] = T(U32ToFloat(point_data0.range(31, 0)));
  x[1] = T(U32ToFloat(point_data0.range(63, 32)));
  x[2] = T(U32ToFloat(point_data1.range(31, 0)));
}

// Read a 1D tensor from a DDR memory
template <typename T, int D0>
void ReadTensor1dNaive(T tensor[D0],
                       const float* src,
                       const int offset)
{
#pragma HLS INLINE off

  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE II=1
    tensor[i] = T(src[offset + i]);
  }
}

// Read a 1D tensor from a DDR memory
template <typename T, int D0>
void ReadTensor1dOpt1(T tensor[D0],
                      const ap_uint<64>* src,
                      const int offset)
{
#pragma HLS INLINE off

  static_assert(D0 % 2 == 0, "`D0` must be a multiple of 2");
  assert(offset % 2 == 0);

  constexpr const int D0Over2 = D0 / 2;
  const int offset2 = offset / 2;

  for (int i = 0; i < D0Over2; ++i) {
#pragma HLS PIPELINE II=1
    const ap_uint<64> tensor_data = src[offset2 + i];
    tensor[i * 2 + 0] = T(U32ToFloat(tensor_data.range(31, 0)));
    tensor[i * 2 + 1] = T(U32ToFloat(tensor_data.range(63, 32)));
  }
}

// Read a 2D tensor from a DDR memory
template <typename T, int D0, int D1>
void ReadTensor2dNaive(T tensor[D0][D1],
                       const float* src,
                       const int offset)
{
#pragma HLS INLINE off

  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE II=1
      const int idx = i * D1 + j;
      tensor[i][j] = T(src[offset + idx]);
    }
  }
}

// Read a 2D tensor from a DDR memory
template <typename T, int D0, int D1>
void ReadTensor2dOpt1(T tensor[D0][D1],
                      const ap_uint<64>* src,
                      const int offset)
{
#pragma HLS INLINE off

  static_assert(D1 % 2 == 0, "`D1` must be a multiple of 2");
  assert(offset % 2 == 0);

  constexpr const int D1Over2 = D1 / 2;
  const int offset2 = offset / 2;

  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1Over2; ++j) {
#pragma HLS PIPELINE II=1
      const int idx = i * D1Over2 + j;
      const ap_uint<64> tensor_data = src[offset2 + idx];
      tensor[i][j * 2 + 0] = T(U32ToFloat(tensor_data.range(31, 0)));
      tensor[i][j * 2 + 1] = T(U32ToFloat(tensor_data.range(63, 32)));
    }
  }
}

// Read a 2D tensor of size (`D0`, 3) from a DDR memory
template <typename T, int D0, int D1>
void ReadTensor2dOpt2(T tensor[D0][D1],
                      const ap_uint<64>* src,
                      const int offset)
{
#pragma HLS INLINE off

  static_assert(D0 % 2 == 0, "`D0` must be a multiple of 2");
  static_assert(D1 == 3, "`D1` must be 3");
  assert(offset % 2 == 0);

  constexpr const int Iter = D0 * D1 / (2 * 3);
  const int offset2 = offset / 2;

  for (int i = 0; i < Iter; ++i) {
#pragma HLS PIPELINE
    const int src_idx = i * 3;
    const int dst_idx = i * 2;
    const ap_uint<128> tensor_data0 = src[offset2 + src_idx + 0];
    const ap_uint<128> tensor_data1 = src[offset2 + src_idx + 1];
    const ap_uint<128> tensor_data2 = src[offset2 + src_idx + 2];
    tensor[dst_idx + 0][0] = T(U32ToFloat(tensor_data0.range(31, 0)));
    tensor[dst_idx + 0][1] = T(U32ToFloat(tensor_data0.range(63, 32)));
    tensor[dst_idx + 0][2] = T(U32ToFloat(tensor_data1.range(31, 0)));
    tensor[dst_idx + 1][0] = T(U32ToFloat(tensor_data1.range(63, 32)));
    tensor[dst_idx + 1][1] = T(U32ToFloat(tensor_data2.range(31, 0)));
    tensor[dst_idx + 1][2] = T(U32ToFloat(tensor_data2.range(63, 32)));
  }
}

// Write a 1D tensor to a DDR memory
template <typename T, int D0>
void WriteTensor1dNaive(float* dst,
                        const T tensor[D0],
                        const int offset)
{
#pragma HLS INLINE off

  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE II=1
    dst[offset + i] = static_cast<float>(tensor[i]);
  }
}

// Write a 1D tensor to a DDR memory
template <typename T, int D0>
void WriteTensor1dOpt1(ap_uint<64>* dst,
                       const T tensor[D0],
                       const int offset)
{
#pragma HLS INLINE off

  static_assert(D0 % 2 == 0, "`D0` must be a multiple of 2");
  assert(offset % 2 == 0);

  constexpr const int D0Over2 = D0 / 2;
  const int offset2 = offset / 2;

  for (int i = 0; i < D0Over2; ++i) {
#pragma HLS PIPELINE II=1
    ap_uint<64> tensor_data;
    tensor_data.range(31, 0) = FloatToU32(
      static_cast<float>(tensor[i * 2 + 0]));
    tensor_data.range(63, 32) = FloatToU32(
      static_cast<float>(tensor[i * 2 + 1]));
    dst[offset2 + i] = tensor_data;
  }
}

// Write a 2D tensor to a DDR memory
template <typename T, int D0, int D1>
void WriteTensor2dNaive(float* dst,
                        const T tensor[D0][D1],
                        const int offset)
{
#pragma HLS INLINE off

  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE II=1
      const int idx = i * D1 + j;
      dst[offset + i] = static_cast<float>(tensor[i][j]);
    }
  }
}

// Naive implementation of the parameter initialization
// Read the parameters for a linear layer from a DDR memory and
// store them to BRAM buffers
// `T` is the type for parameters
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
template <typename T, int InDims, int OutDims>
void ReadLinearParamsNaive(LinearParams<T, InDims, OutDims>* linear,
                           const float* params,
                           const int offset)
{
  // `params` contains weight parameters of size (`OutDims`, `InDims`) and
  // bias parameters of size (`OutDims`) in a contiguous buffer

#pragma HLS INLINE off

  // Read the weight
  ReadTensor2dNaive<T, OutDims, InDims>(linear->weight, params, offset);
  // Read the bias
  ReadTensor1dNaive<T, OutDims>(linear->bias, params,
                                offset + InDims * OutDims);
}

// Parallel implementation of the parameter initialization
// Read the parameters for a linear layer from a DDR memory and
// store them to BRAM buffers
// `T` is the type for parameters
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
template <typename T, int InDims, int OutDims>
void ReadLinearParamsOpt1(LinearParams<T, InDims, OutDims>* linear,
                          const ap_uint<64>* params,
                          const int offset)
{
#pragma HLS INLINE
  // `params` contains weight parameters of size (`OutDims`, `InDims`) and
  // bias parameters of size (`OutDims`) in a contiguous buffer

  static_assert(InDims % 2 == 0, "`InDims` must be a multiple of 2");
  static_assert(OutDims % 2 == 0, "`OutDims` must be a multiple of 2");
  assert(offset % 2 == 0);

  // Read the weight
  ReadTensor2dOpt1<T, OutDims, InDims>(linear->weight, params, offset);
  // Read the bias
  ReadTensor1dOpt1<T, OutDims>(linear->bias, params,
                               offset + InDims * OutDims);
}

// Naive implementation of the parameter initialization
// Read the parameters for a 1D batch normalization layer from a DDR memory and
// store them to BRAM buffers
// `T` is the type for parameters
// `Dims` is the number of input and output dimensions
template <typename T, int Dims>
void ReadBatchNorm1dParamsNaive(BatchNorm1dParams<T, Dims>* bn,
                                const float* params,
                                const int offset)
{
  // `params` contains scale parameters of size (`Dims`),
  // bias of size (`Dims`), and mean of size (`Dims`) in a contiguous buffer

#pragma HLS INLINE off

  // Read the scale
  ReadTensor1dNaive<T, Dims>(bn->scale, params, offset);
  // Read the bias
  ReadTensor1dNaive<T, Dims>(bn->bias, params, offset + Dims);
  // Read the mean
  ReadTensor1dNaive<T, Dims>(bn->mean, params, offset + Dims * 2);
}

// Parallel implementation of the parameter initialization
// Read the parameters for a 1D batch normalization layer from a DDR memory and
// store them to BRAM buffers
// `T` is the type for parameters
// `Dims` is the number of input and output dimensions
template <typename T, int Dims>
void ReadBatchNorm1dParamsOpt1(BatchNorm1dParams<T, Dims>* bn,
                               const ap_uint<64>* params,
                               const int offset)
{
#pragma HLS INLINE
  // `params` contains scale parameters of size (`Dims`),
  // bias of size (`Dims`), and mean of size (`Dims`) in a contiguous buffer

  static_assert(Dims % 2 == 0, "`Dims` must be a multiple of 2");
  assert(offset % 2 == 0);

  // Read the scale
  ReadTensor1dOpt1<T, Dims>(bn->scale, params, offset);
  // Read the bias
  ReadTensor1dOpt1<T, Dims>(bn->bias, params, offset + Dims);
  // Read the mean
  ReadTensor1dOpt1<T, Dims>(bn->mean, params, offset + Dims * 2);
}

// Naive implementation of the parameter initialization
// Read the parameters for a linear and 1D batch normalization layer
// from a DDR memory and store them to BRAM buffers
// `T` is the type for parameters
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
template <typename T, int InDims, int OutDims>
void ReadBlockParamsNaive(LinearParams<T, InDims, OutDims>* linear,
                          BatchNorm1dParams<T, OutDims>* bn,
                          const float* params)
{
#pragma HLS INLINE off

  // Read the weight for a linear layer
  ReadTensor2dNaive<T, OutDims, InDims>(linear->weight, params, 0);
  // Read the bias for a linear layer
  ReadTensor1dNaive<T, OutDims>(linear->bias, params, InDims * OutDims);
  // Read the scale for a linear layer
  ReadTensor1dNaive<T, OutDims>(bn->scale, params,
                                InDims * OutDims + OutDims);
  // Read the bias for a linear layer
  ReadTensor1dNaive<T, OutDims>(bn->bias, params,
                                InDims * OutDims + OutDims * 2);
  // Read the mean for a linear layer
  ReadTensor1dNaive<T, OutDims>(bn->mean, params,
                                InDims * OutDims + OutDims * 3);
}

// Parallel implementation of the parameter initialization
// Read the parameters for a linear and 1D batch normalization layer
// from a DDR memory and store them to BRAM buffers
// `T` is the type for parameters
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
template <typename T, int InDims, int OutDims>
void ReadBlockParamsOpt1(LinearParams<T, InDims, OutDims>* linear,
                         BatchNorm1dParams<T, OutDims>* bn,
                         const ap_uint<64>* params)
{
#pragma HLS INLINE

  static_assert(InDims % 2 == 0, "`InDims` must be a multiple of 2");
  static_assert(OutDims % 2 == 0, "`OutDims` must be a multiple of 2");

  // Read the weight for a linear layer
  ReadTensor2dOpt1<T, OutDims, InDims>(linear->weight, params, 0);
  // Read the bias for a linear layer
  ReadTensor1dOpt1<T, OutDims>(linear->bias, params, InDims * OutDims);
  // Read the scale for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->scale, params,
                               InDims * OutDims + OutDims);
  // Read the bias for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->bias, params,
                               InDims * OutDims + OutDims * 2);
  // Read the mean for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->mean, params,
                               InDims * OutDims + OutDims * 3);
}

// Parallel implementation of the parameter initialization
// Read the parameters for a linear and 1D batch normalization layer
// from a DDR memory and store them to BRAM buffers
// `T` is the type for parameters
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
template <typename T, int InDims, int OutDims>
void ReadBlockParamsOpt2(LinearParams<T, InDims, OutDims>* linear,
                         BatchNorm1dParams<T, OutDims>* bn,
                         const ap_uint<64>* params)
{
#pragma HLS INLINE

  static_assert(InDims == 3, "`InDims` must be 3");
  static_assert(OutDims % 2 == 0, "`OutDims` must be a multiple of 2");

  // Read the weight for a linear layer
  ReadTensor2dOpt2<T, OutDims, InDims>(linear->weight, params, 0);
  // Read the bias for a linear layer
  ReadTensor1dOpt1<T, OutDims>(linear->bias, params, InDims * OutDims);
  // Read the scale for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->scale, params,
                               InDims * OutDims + OutDims);
  // Read the bias for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->bias, params,
                               InDims * OutDims + OutDims * 2);
  // Read the mean for a linear layer
  ReadTensor1dOpt1<T, OutDims>(bn->mean, params,
                               InDims * OutDims + OutDims * 3);
}

#endif // DATA_TRANSFER_HPP
