//
// Created by 구현우 on 2023/12/08.
//

#ifndef MY_OPENCL_BASICTRANSFORMERBLOCK_H
#define MY_OPENCL_BASICTRANSFORMERBLOCK_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "LayerNorm.h"
#include "CrossAttention.h"
#include "FeedForward.h"

#include <android/asset_manager_jni.h>
#include "../kernel/unit/LayerNormKernel.h"
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/UtilKernel.h"

class BasicTransformerBlock {
public:
    BasicTransformerBlock(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                          AAssetManager *assetManager,
                          size_t dim, size_t context_dim, size_t headSize, size_t headDim,
                          const std::string &layer_norm_1_weight_name,
                          const std::string &layer_norm_1_bias_name,
                          const std::string &layer_norm_2_weight_name,
                          const std::string &layer_norm_2_bias_name,
                          const std::string &layer_norm_3_weight_name,
                          const std::string &layer_norm_3_bias_name,
                          const std::string &cross_1_q_linear_weight_name,
                          const std::string &cross_1_k_linear_weight_name,
                          const std::string &cross_1_v_linear_weight_name,
                          const std::string &cross_1_out_linear_weight_name,
                          const std::string &cross_1_out_linear_bias_name,
                          const std::string &cross_2_q_linear_weight_name,
                          const std::string &cross_2_k_linear_weight_name,
                          const std::string &cross_2_v_linear_weight_name,
                          const std::string &cross_2_out_linear_weight_name,
                          const std::string &cross_2_out_linear_bias_name,
                          const std::string &ff_geglu_linear_weight_name,
                          const std::string &ff_geglu_linear_bias_name,
                          const std::string &ff_net_linear_weight_name,
                          const std::string &ff_net_linear_bias_name,
                          std::shared_ptr<LayerNormKernel> layerNormKernel,
                          LinearKernel &linearKernel,
                          UtilKernel &utilKernel
    );

    ~BasicTransformerBlock();

    cl_int forward(cl_mem input, cl_mem condition, cl_mem output,
                   cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    cl_command_queue cmdQueue;
    cl_context context;

    UtilKernel utilKernel;

    LayerNorm *layerNorm1;
    LayerNorm *layerNorm2;
    LayerNorm *layerNorm3;

    CrossAttention *crossAttention1;
    CrossAttention *crossAttention2;

    FeedForward *feedForward;
};


#endif //MY_OPENCL_BASICTRANSFORMERBLOCK_H
