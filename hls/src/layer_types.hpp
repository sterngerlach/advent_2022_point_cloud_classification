
// layer_types.hpp

#ifndef LAYER_TYPES_HPP
#define LAYER_TYPES_HPP

// Parameters for fully-connected layers
template <typename T, int InDims_, int OutDims_>
struct LinearParams
{
  enum
  {
    InDims = InDims_,
    OutDims = OutDims_,
  };

  T weight[OutDims][InDims];
  T bias[OutDims];
};

// Parameters for 1D batch normalization layers
template <typename T, int Dims_>
struct BatchNorm1dParams
{
  enum
  {
    Dims = Dims_,
  };

  // `scale` is obtained by multiplying weights and reciprocal of the
  // square root of the standard deviation (to reduce the computational cost)
  T scale[Dims];
  T bias[Dims];
  T mean[Dims];
};

#endif // LAYER_TYPES_HPP
