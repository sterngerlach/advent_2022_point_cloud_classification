
// top_naive.cpp

#include "data_transfer.hpp"
#include "data_type_config.hpp"
#include "inference_cls_naive.hpp"
#include "inference_feat_naive.hpp"
#include "net_params.hpp"
#include "op_modes.hpp"

void PointNetClsTop(const int op_mode,
                    const float* point_cloud,
                    const int num_points,
                    float* out_logits,
                    const float* feat_params1,
                    const float* feat_params2,
                    const float* feat_params3,
                    const float* feat_params4,
                    const float* feat_params5,
                    const float* cls_params1,
                    const float* cls_params2,
                    const float* cls_params3)
{
#pragma HLS INTERFACE m_axi port=point_cloud offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out_logits offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=feat_params1 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=feat_params2 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=feat_params3 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=feat_params4 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=feat_params5 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cls_params1 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cls_params2 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cls_params3 offset=slave bundle=gmem0

#pragma HLS INTERFACE s_axilite port=op_mode bundle=control
#pragma HLS INTERFACE s_axilite port=point_cloud bundle=control
#pragma HLS INTERFACE s_axilite port=num_points bundle=control
#pragma HLS INTERFACE s_axilite port=out_logits bundle=control
#pragma HLS INTERFACE s_axilite port=feat_params1 bundle=control
#pragma HLS INTERFACE s_axilite port=feat_params2 bundle=control
#pragma HLS INTERFACE s_axilite port=feat_params3 bundle=control
#pragma HLS INTERFACE s_axilite port=feat_params4 bundle=control
#pragma HLS INTERFACE s_axilite port=feat_params5 bundle=control
#pragma HLS INTERFACE s_axilite port=cls_params1 bundle=control
#pragma HLS INTERFACE s_axilite port=cls_params2 bundle=control
#pragma HLS INTERFACE s_axilite port=cls_params3 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Parameters for feature extraction
  LinearParams<param_t, kFeatDims0, kFeatDims1> feat_conv1;
  LinearParams<param_t, kFeatDims1, kFeatDims2> feat_conv2;
  LinearParams<param_t, kFeatDims2, kFeatDims3> feat_conv3;
  LinearParams<param_t, kFeatDims3, kFeatDims4> feat_conv4;
  LinearParams<param_t, kFeatDims4, kFeatDims5> feat_conv5;
  BatchNorm1dParams<param_t, kFeatDims1> feat_bn1;
  BatchNorm1dParams<param_t, kFeatDims2> feat_bn2;
  BatchNorm1dParams<param_t, kFeatDims3> feat_bn3;
  BatchNorm1dParams<param_t, kFeatDims4> feat_bn4;
  BatchNorm1dParams<param_t, kFeatDims5> feat_bn5;

  // Parameters for classification network
  // LinearParams<param_t, kClsDims0, kClsDims1> cls_fc1;
  // LinearParams<param_t, kClsDims1, kClsDims2> cls_fc2;
  LinearParams<param_t, kClsDims2, kClsDims3> cls_fc3;
  BatchNorm1dParams<param_t, kClsDims1> cls_bn1;
  BatchNorm1dParams<param_t, kClsDims2> cls_bn2;

  // Extracted feature
  value_t feature[kFeatDims5];

  if (op_mode == kModeInitWeights) {
    // Initialize the PointNet feature extraction network
    InitializeFeatNaive<param_t>(
      &feat_conv1, &feat_conv2, &feat_conv3, &feat_conv4, &feat_conv5,
      &feat_bn1, &feat_bn2, &feat_bn3, &feat_bn4, &feat_bn5,
      feat_params1, feat_params2, feat_params3, feat_params4, feat_params5);
    // Initialize the classification network
    InitializeClsNaive<param_t>(
      &cls_fc3, &cls_bn1, &cls_bn2,
      cls_params1, cls_params2, cls_params3);
  } else if (op_mode == kModeInference) {
    // Run the PointNet feature extraction
    InferenceFeatNaive<value_t, param_t, 1024>(
      point_cloud, num_points, feature,
      &feat_conv1, &feat_conv2, &feat_conv3, &feat_conv4, &feat_conv5,
      &feat_bn1, &feat_bn2, &feat_bn3, &feat_bn4, &feat_bn5);

    // Run the classification
    InferenceClsNaive<value_t, param_t>(
      feature, out_logits,
      &cls_fc3, &cls_bn1, &cls_bn2,
      cls_params1, cls_params2, cls_params3);
  }
}
