
// inference_feat_opt2.hpp

#ifndef INFERENCE_FEAT_OPT2_HPP
#define INFERENCE_FEAT_OPT2_HPP

#include "batch_norm_1d.hpp"
#include "data_transfer.hpp"
#include "layer_types.hpp"
#include "linear.hpp"
#include "max_pool_1d.hpp"
#include "net_params.hpp"
#include "vector_utils.hpp"

// Naive implementation of the parameter initialization
// Same as the naive implementation
// `T` is the type for parameters
template <typename T>
void InitializeFeatOpt2(LinearParams<T, kFeatDims0, kFeatDims1>* conv1,
                        LinearParams<T, kFeatDims1, kFeatDims2>* conv2,
                        LinearParams<T, kFeatDims2, kFeatDims3>* conv3,
                        LinearParams<T, kFeatDims3, kFeatDims4>* conv4,
                        LinearParams<T, kFeatDims4, kFeatDims5>* conv5,
                        BatchNorm1dParams<T, kFeatDims1>* bn1,
                        BatchNorm1dParams<T, kFeatDims2>* bn2,
                        BatchNorm1dParams<T, kFeatDims3>* bn3,
                        BatchNorm1dParams<T, kFeatDims4>* bn4,
                        BatchNorm1dParams<T, kFeatDims5>* bn5,
                        const float* params1,
                        const float* params2,
                        const float* params3,
                        const float* params4,
                        const float* params5)
{
#pragma HLS INLINE off

  ReadBlockParamsNaive<T, kFeatDims0, kFeatDims1>(conv1, bn1, params1);
  ReadBlockParamsNaive<T, kFeatDims1, kFeatDims2>(conv2, bn2, params2);
  ReadBlockParamsNaive<T, kFeatDims2, kFeatDims3>(conv3, bn3, params3);
  ReadBlockParamsNaive<T, kFeatDims3, kFeatDims4>(conv4, bn4, params4);
  ReadBlockParamsNaive<T, kFeatDims4, kFeatDims5>(conv5, bn5, params5);
}

// Parallel implementation of the PointNet feature extraction
// `T` is the type for layer input, output, and intermediate results
// `U` is the type for parameters
// `N` is the expected number of input points (e.g., 1024)
template <typename T, typename U, int N>
void InferenceFeatOpt2(const float* point_cloud,
                       const int num_points,
                       T feature[kFeatDims5],
                       const LinearParams<U, kFeatDims0, kFeatDims1>* conv1,
                       const LinearParams<U, kFeatDims1, kFeatDims2>* conv2,
                       const LinearParams<U, kFeatDims2, kFeatDims3>* conv3,
                       const LinearParams<U, kFeatDims3, kFeatDims4>* conv4,
                       const LinearParams<U, kFeatDims4, kFeatDims5>* conv5,
                       const BatchNorm1dParams<U, kFeatDims1>* bn1,
                       const BatchNorm1dParams<U, kFeatDims2>* bn2,
                       const BatchNorm1dParams<U, kFeatDims3>* bn3,
                       const BatchNorm1dParams<U, kFeatDims4>* bn4,
                       const BatchNorm1dParams<U, kFeatDims5>* bn5)
{
#pragma HLS INLINE off

  // Zero-initialize the output feature
  VectorNdSetZero<T, kFeatDims5>(feature);

  // Compute the feature
  for (int i = 0; i < num_points; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N avg=N
#pragma HLS LOOP_FLATTEN off
#pragma HLS DATAFLOW

#pragma HLS STABLE variable=point_cloud
#pragma HLS STABLE variable=num_points
#pragma HLS STABLE variable=feature
#pragma HLS STABLE variable=conv1
#pragma HLS STABLE variable=conv2
#pragma HLS STABLE variable=conv3
#pragma HLS STABLE variable=conv4
#pragma HLS STABLE variable=conv5
#pragma HLS STABLE variable=bn1
#pragma HLS STABLE variable=bn2
#pragma HLS STABLE variable=bn3
#pragma HLS STABLE variable=bn4
#pragma HLS STABLE variable=bn5

    // Input, output, and intermediate results
    T x0[kFeatDims0];
    T x1[kFeatDims1];
    T x2[kFeatDims1];
    T x3[kFeatDims2];
    T x4[kFeatDims2];
    T x5[kFeatDims3];
    T x6[kFeatDims3];
    T x7[kFeatDims4];
    T x8[kFeatDims4];
    T x9[kFeatDims5];
    T x10[kFeatDims5];

#pragma HLS ARRAY_PARTITION variable=x3 type=cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=x5 type=cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=x7 type=cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=x9 type=cyclic factor=64 dim=1

    // Read a point from a DDR memory
    ReadPointNaive<T>(point_cloud, i, x0);

    // Compute a point feature
    LinearOpt1<T, U, kFeatDims0, kFeatDims1, false, 2>(
      x0, x1, conv1->weight, conv1->bias);
    BatchNorm1dReLUOpt1<T, U, kFeatDims1, 2>(
      x1, x2, bn1->scale, bn1->bias, bn1->mean);
    LinearOpt1<T, U, kFeatDims1, kFeatDims2, false, 8>(
      x2, x3, conv2->weight, conv2->bias);
    BatchNorm1dReLUOpt1<T, U, kFeatDims2, 2>(
      x3, x4, bn2->scale, bn2->bias, bn2->mean);
    LinearOpt1<T, U, kFeatDims2, kFeatDims3, false, 8>(
      x4, x5, conv3->weight, conv3->bias);
    BatchNorm1dReLUOpt1<T, U, kFeatDims3, 2>(
      x5, x6, bn3->scale, bn3->bias, bn3->mean);
    LinearOpt1<T, U, kFeatDims3, kFeatDims4, false, 16>(
      x6, x7, conv4->weight, conv4->bias);
    BatchNorm1dReLUOpt1<T, U, kFeatDims4, 2>(
      x7, x8, bn4->scale, bn4->bias, bn4->mean);
    LinearOpt1<T, U, kFeatDims4, kFeatDims5, false, 128>(
      x8, x9, conv5->weight, conv5->bias);
    BatchNorm1dReLUOpt1<T, U, kFeatDims5, 2>(
      x9, x10, bn5->scale, bn5->bias, bn5->mean);

    // Update the output feature
    MaxPool1dOpt1<T, kFeatDims5, 2>(x10, feature);
  }
}

#endif // INFERENCE_FEAT_OPT2_HPP
