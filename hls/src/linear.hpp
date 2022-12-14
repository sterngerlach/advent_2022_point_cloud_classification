
// linear.hpp

#ifndef LINEAR_HPP
#define LINEAR_HPP

// Naive implementation of the fully-connected layer
// `T` is the type for values
// `TParam` is the type for weight and bias
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
// `ApplyReLU` is the flag to apply ReLU activation
template <typename T, typename TParam,
          int InDims, int OutDims, bool ApplyReLU>
void LinearNaive(const T x[InDims],
                 T y[OutDims],
                 const TParam weight[OutDims][InDims],
                 const TParam bias[OutDims])
{
  // `x` is of size (1, `InDims`)
  // `weight` is of size (`OutDims`, `InDims`)
  // `bias` is of size (`OutDims`)
  // `y` is of size (1, `OutDims`)

#pragma HLS INLINE off

  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE off
    T val = bias[i];

    for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE
      val += x[j] * weight[i][j];
    }

    if (ApplyReLU)
      y[i] = val > T(0) ? val : T(0);
    else
      y[i] = val;
  }
}

// Naive implementation of the fully-connected layer
// Weight and bias parameters are stored on the DDR memory
// `T` is the type for values
// `TParam` is the type for weight and bias
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
// `ApplyReLU` is the flag to apply ReLU activation
template <typename T, typename TParam,
          int InDims, int OutDims, bool ApplyReLU>
void LinearNaiveDDR(const T x[InDims],
                    T y[OutDims],
                    const float* params,
                    const int offset)
{
  // `x` is of size (1, `InDims`)
  // `y` is of size (1, `OutDims`)
  // `params` contains weight parameters of size (`OutDims`, `InDims`) and
  // bias parameters of size (`OutDims`) in a contiguous buffer

#pragma HLS INLINE off

  constexpr const int OffsetToBias = OutDims * InDims;

  TParam bias[OutDims];

  // Copy the bias parameters in advance
  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE II=1
    bias[i] = TParam(params[offset + OffsetToBias + i]);
  }

  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE off
    T val = bias[i];

    TParam weight[InDims];

    for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE II=1
      weight[j] = TParam(params[offset + i * InDims + j]);
    }

    for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE
      val += x[j] * weight[j];
    }

    if (ApplyReLU)
      y[i] = val > T(0) ? val : T(0);
    else
      y[i] = val;
  }
}

// Parallel implementation of the fully-connected layer
// Matrix-vector multiplication is parallelized along the output dimension
// `T` is the type for values
// `TParam` is the type for weight and bias
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
// `ApplyReLU` is the flag to apply ReLU activation
// `B` is the block size for the output dimension
template <typename T, typename TParam,
          int InDims, int OutDims, bool ApplyReLU, int B>
void LinearOpt1(const T x[InDims],
                T y[OutDims],
                const TParam weight[OutDims][InDims],
                const TParam bias[OutDims])
{
  // `x` is of size (1, `InDims`)
  // `weight` is of size (`OutDims`, `InDims`)
  // `bias` is of size (`OutDims`)
  // `y` is of size (1, `OutDims`)

#pragma HLS INLINE off

  // `OutDims` must be a multiple of `B`
  static_assert(OutDims % B == 0, "`OutDims` must be a multiple of `B`");

  for (int i0 = 0; i0 < OutDims; i0 += B) {
#pragma HLS PIPELINE off
    T vals[B];
#pragma HLS ARRAY_PARTITION variable=vals type=complete dim=1

    for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE
      for (int i1 = 0; i1 < B; ++i1) {
#pragma HLS UNROLL
        int i = i0 + i1;
        T last = (j == 0) ? T(bias[i]) : vals[i1];
        vals[i1] = last + x[j] * weight[i][j];
      }
    }

    for (int i1 = 0; i1 < B; ++i1) {
#pragma HLS UNROLL
      int i = i0 + i1;
      if (ApplyReLU)
        y[i] = vals[i1] > T(0) ? vals[i1] : T(0);
      else
        y[i] = vals[i1];
    }
  }
}

// Parallel implementation of the fully-connected layer
// Weight and bias parameters are stored on the DDR memory
// Matrix-vector multiplication is parallelized along the output dimension
// `T` is the type for values
// `TParam` is the type for weight and bias
// `InDims` is the number of input dimensions
// `OutDims` is the number of output dimensions
// `ApplyReLU` is the flag to apply ReLU activation
// `B` is the block size for the output dimension
template <typename T, typename TParam,
          int InDims, int OutDims, bool ApplyReLU, int B>
void LinearOpt1DDR(const T x[InDims],
                   T y[OutDims],
                   const float* params,
                   const int offset)
{
  // `x` is of size (1, `InDims`)
  // `y` is of size (1, `OutDims`)
  // `params` contains weight parameters of size (`OutDims`, `InDims`) and
  // bias parameters of size (`OutDims`) in a contiguous buffer

#pragma HLS INLINE off

  // `OutDims` must be a multiple of `B`
  static_assert(OutDims % B == 0, "`OutDims` must be a multiple of `B`");
  // `B` must be larger than 1
  static_assert(B > 1, "`B` must be larger than 1");

  constexpr const int BHalf = B / 2;
  constexpr const int OffsetToBias = OutDims * InDims;

  TParam bias[OutDims];
#pragma HLS ARRAY_PARTITION variable=bias type=cyclic factor=BHalf dim=1

  // Copy the bias parameters in advance
  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE II=1
    bias[i] = TParam(params[offset + OffsetToBias + i]);
  }

  for (int i0 = 0; i0 < OutDims; i0 += B) {
#pragma HLS PIPELINE off
    T vals[B];
#pragma HLS ARRAY_PARTITION variable=vals type=complete dim=1
    TParam weight[B][InDims];
#pragma HLS ARRAY_PARTITION variable=weight type=cyclic factor=BHalf dim=1

    // Copy the weight parameters for `B` outputs
    const int offset0 = offset + i0 * InDims;
    for (int i1 = 0; i1 < B; ++i1) {
      for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE II=1
        weight[i1][j] = TParam(params[offset0 + i1 * InDims + j]);
      }
    }

    for (int j = 0; j < InDims; ++j) {
#pragma HLS PIPELINE
      for (int i1 = 0; i1 < B; ++i1) {
#pragma HLS UNROLL
        int i = i0 + i1;
        if (i < OutDims) {
          T last = (j == 0) ? T(bias[i]) : vals[i1];
          vals[i1] = last + x[j] * weight[i1][j];
        }
      }
    }

    for (int i1 = 0; i1 < B; ++i1) {
#pragma HLS UNROLL
      int i = i0 + i1;
      if (i < OutDims) {
        if (ApplyReLU)
          y[i] = vals[i1] > T(0) ? vals[i1] : T(0);
        else
          y[i] = vals[i1];
      }
    }
  }
}

#endif // LINEAR_HPP
