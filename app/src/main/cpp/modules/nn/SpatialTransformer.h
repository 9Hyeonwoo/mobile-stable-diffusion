//
// Created by 구현우 on 2023/12/08.
//

#ifndef MY_OPENCL_SPATIALTRANSFORMER_H
#define MY_OPENCL_SPATIALTRANSFORMER_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include "GroupNorm.h"
#include "Linear.h"
#include "BasicTransformerBlock.h"
#include "../kernel/unit/LayerNormKernel.h"
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/UtilKernel.h"
#include "../kernel/unit/CrossAttentionKernel.h"
#include "../kernel/unit/GEGLUKernel.h"
#include "../kernel/unit/GroupNormKernel.h"

class SpatialTransformer {
public:
    SpatialTransformer(
            cl_context context, cl_command_queue cmdQueue,
            size_t channels, size_t context_dim, size_t headSize, size_t headDim,
            const std::string &group_norm_weight_name,
            const std::string &group_norm_bias_name,
            const std::string &in_linear_weight_name,
            const std::string &in_linear_bias_name,
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
            const std::string &out_linear_weight_name,
            const std::string &out_linear_bias_name,
            std::shared_ptr<LayerNormKernel> layerNormKernel,
            std::shared_ptr<LinearKernel> linearKernel,
            std::shared_ptr<UtilKernel> utilKernel,
            std::shared_ptr<CrossAttentionKernel> crossAttentionKernel,
            std::shared_ptr<GEGLUKernel> gegluKernel,
            std::shared_ptr<GroupNormKernel> groupNormKernel
    );

    ~SpatialTransformer();

    cl_int forward(cl_mem input, cl_mem condition, cl_mem output,
                   cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    size_t channels;
    cl_command_queue cmdQueue;
    cl_context context;

    GroupNorm *groupNorm;
    Linear *projInLinear;
    BasicTransformerBlock *transformer;
    Linear *projOutLinear;

    std::shared_ptr<UtilKernel> utilKernel;
};


#endif //MY_OPENCL_SPATIALTRANSFORMER_H
