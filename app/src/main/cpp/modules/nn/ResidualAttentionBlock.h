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
#include <android/asset_manager_jni.h>

class ResidualAttentionBlock {
public:
    ResidualAttentionBlock(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                           AAssetManager *assetManager, size_t numHeads);

    ~ResidualAttentionBlock();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    LayerNorm *ln_1;
    LayerNorm *ln_2;
    MultiHeadAttention *attn;
    Linear *mlp_c_fc;

    cl_context context;
    cl_command_queue cmdQueue;
    AAssetManager *assetManager;

    cl_kernel kernel_elemwise_add;
    cl_kernel kernel_gelu;
};


#endif //MY_OPENCL_RESIDUALATTENTIONBLOCK_H
