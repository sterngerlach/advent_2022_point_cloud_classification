
// data_transfer.hpp

#ifndef DATA_TRANSFER_HPP
#define DATA_TRANSFER_HPP

#include "layer_types.hpp"

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

#endif // DATA_TRANSFER_HPP
