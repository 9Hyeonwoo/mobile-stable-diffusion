//
// Created by 구현우 on 2023/12/02.
//

#ifndef MY_OPENCL_RESIDUALATTENTIONBLOCK_H
#define MY_OPENCL_RESIDUALATTENTIONBLOCK_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiHeadAttention.h"
#include "../kernel/unit/LayerNormKernel.h"
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/MultiHeadAttentionKernel.h"
#include "../kernel/unit/UtilKernel.h"

class ResidualAttentionBlock {
public:
    ResidualAttentionBlock(cl_context context, cl_command_queue cmdQueue,
                           size_t d_model, size_t numHeads,
                           const std::string &ln_1_weight_name,
                           const std::string &ln_1_bias_name,
                           const std::string &ln_2_weight_name,
                           const std::string &ln_2_bias_name,
                           const std::string &attn_in_proj_weight_name,
                           const std::string &attn_in_proj_bias_name,
                           const std::string &attn_out_proj_weight_name,
                           const std::string &attn_out_proj_bias_name,
                           const std::string &mlp_c_fc_weight_name,
                           const std::string &mlp_c_fc_bias_name,
                           const std::string &mlp_c_proj_weight_name,
                           const std::string &mlp_c_proj_bias_name,
                           cl_mem attentionMask,
                           std::shared_ptr<LayerNormKernel> layerNormKernel,
                           std::shared_ptr<LinearKernel> linearKernel,
                           std::shared_ptr<MultiHeadAttentionKernel> multiHeadAttentionKernel,
                           std::shared_ptr<UtilKernel> utilKernel
    );

    ~ResidualAttentionBlock();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    LayerNorm *ln_1;
    LayerNorm *ln_2;
    MultiHeadAttention *attn;
    Linear *mlp_c_fc;
    Linear *mlp_c_proj;

    cl_context context;
    cl_command_queue cmdQueue;

    std::shared_ptr<UtilKernel> utilKernel;
};


#endif //MY_OPENCL_RESIDUALATTENTIONBLOCK_H
