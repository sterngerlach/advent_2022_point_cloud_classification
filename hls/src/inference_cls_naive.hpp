
// inference_cls_naive.hpp

#ifndef INFERENCE_CLS_NAIVE_HPP
#define INFERENCE_CLS_NAIVE_HPP

#include "batch_norm_1d.hpp"
#include "data_transfer.hpp"
#include "layer_types.hpp"
#include "linear.hpp"
#include "net_params.hpp"
#include "vector_utils.hpp"

// Naive implementation of the parameter initialization
// `T` is the type for parameters
template <typename T>
void InitializeClsNaive(LinearParams<T, kClsDims2, kClsDims3>* fc3,
                        BatchNorm1dParams<T, kClsDims1>* bn1,
                        BatchNorm1dParams<T, kClsDims2>* bn2,
                        const float* params1,
                        const float* params2,
                        const float* params3)
{
#pragma HLS INLINE off

  ReadBatchNorm1dParamsNaive<T, kClsDims1>(
    bn1, params1, kClsDims0 * kClsDims1 + kClsDims1);
  ReadBatchNorm1dParamsNaive<T, kClsDims2>(
    bn2, params2, kClsDims1 * kClsDims2 + kClsDims2);
  ReadLinearParamsNaive<T, kClsDims2, kClsDims3>(
    fc3, params3, 0);
}

// Naive implementation of the classification network
// `T` is the type for layer input, output, and intermediate results
// `U` is the type for parameters
template <typename T, typename U>
void InferenceClsNaive(const T feature[kFeatDims5],
                       float* out_logits,
                       const LinearParams<U, kClsDims2, kClsDims3>* fc3,
                       const BatchNorm1dParams<U, kClsDims1>* bn1,
                       const BatchNorm1dParams<U, kClsDims2>* bn2,
                       const float* params1,
                       const float* params2,
                       const float* params3)
{
#pragma HLS INLINE off

  static_assert(kFeatDims5 == kClsDims0,
                "Feature dimension should be equal to the input dimension");

  // Input, output, and intermediate results
  T x0[kClsDims1];
  T x1[kClsDims1];
  T x2[kClsDims2];
  T x3[kClsDims2];
  T x4[kClsDims3];

  // Compute logits
  LinearNaiveDDR<T, U, kClsDims0, kClsDims1, false>(
    feature, x0, params1, 0);
  BatchNorm1dReLUNaive<T, U, kClsDims1>(
    x0, x1, bn1->scale, bn1->bias, bn1->mean);
  LinearNaiveDDR<T, U, kClsDims1, kClsDims2, false>(
    x1, x2, params2, 0);
  BatchNorm1dReLUNaive<T, U, kClsDims2>(
    x2, x3, bn2->scale, bn2->bias, bn2->mean);
  LinearNaive<T, U, kClsDims2, kClsDims3, false>(
    x3, x4, fc3->weight, fc3->bias);

  // Write the result
  WriteTensor1dNaive<T, kClsDims3>(out_logits, x4, 0);
}

#endif // INFERENCE_CLS_NAIVE_HPP
