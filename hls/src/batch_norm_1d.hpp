
// batch_norm_1d.hpp

#ifndef BATCH_NORM_1D_HPP
#define BATCH_NORM_1D_HPP

// Naive implementation of the 1D batch normalization and ReLU activation
// `T` is the type for values
// `TParam` is the type for parameters
// `Dims` is the number of input and output dimensions
template <typename T, typename TParam, int Dims>
void BatchNorm1dReLUNaive(const T x[Dims],
                          T y[Dims],
                          const TParam scale[Dims],
                          const TParam bias[Dims],
                          const TParam mean[Dims])
{
  // `x` is of size (1, `Dims`)
  // `y` is of size (1, `Dims`)
  // `scale` is of size (1, `Dims`)
  // `bias` is of size (1, `Dims`)
  // `mean` is of size (1, `Dims`)

  // `scale` is the multiplication of the weight and reciprocal of the
  // standard deviation (to reduce the on-chip memory consumption)

#pragma HLS INLINE off

  for (int i = 0; i < Dims; ++i) {
#pragma HLS PIPELINE
    // Batch normalization with the learned parameters
    T val = (x[i] - mean[i]) * scale[i] + bias[i];
    // ReLU activation
    y[i] = val > T(0) ? val : T(0);
  }
}

// Parallel implementation of the 1D batch normalization and ReLU activation
// `T` is the type for values
// `TParam` is the type for parameters
// `Dims` is the number of input and output dimensions
// `B` is the block size for the output dimension
template <typename T, typename TParam, int Dims, int B>
void BatchNorm1dReLUOpt1(const T x[Dims],
                         T y[Dims],
                         const TParam scale[Dims],
                         const TParam bias[Dims],
                         const TParam mean[Dims])
{
  // `x` is of size (1, `Dims`)
  // `y` is of size (1, `Dims`)
  // `scale` is of size (1, `Dims`)
  // `bias` is of size (1, `Dims`)
  // `mean` is of size (1, `Dims`)

  // `scale` is the multiplication of the weight and reciprocal of the
  // standard deviation (to reduce the on-chip memory consumption)

#pragma HLS INLINE off

  static_assert(Dims % B == 0, "`Dims` must be a multiple of `B`");

  for (int i0 = 0; i0 < Dims; i0 += B) {
#pragma HLS PIPELINE
    for (int i1 = 0; i1 < B; ++i1) {
#pragma HLS UNROLL
      int i = i0 + i1;
      // Batch normalization with the learned parameters
      T val = (x[i] - mean[i]) * scale[i] + bias[i];
      // ReLU activation
      y[i] = val > T(0) ? val : T(0);
    }
  }
}

#endif // BATCH_NORM_1D_HPP
