//
// Created by 구현우 on 2023/12/02.
//

#ifndef MY_OPENCL_MULTIHEADATTENTION_H
#define MY_OPENCL_MULTIHEADATTENTION_H

#define CL_TARGET_OPENCL_VERSION 200
#include "CL/opencl.h"
#include "Linear.h"
#include <android/asset_manager_jni.h>

class MultiHeadAttention {
public:
    MultiHeadAttention(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                       AAssetManager *assetManager, size_t embed_dim, size_t numHeads,
                       const std::string &in_proj_weight_name,
                       const std::string &in_proj_bias_name,
                       const std::string &out_proj_weight_name,
                       const std::string &out_proj_bias_name);

    ~MultiHeadAttention();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();
private:
    cl_context context;
    cl_command_queue cmdQueue;

    size_t numHeads;

    Linear *attnInProj0;
    Linear *attnOutProj0;

    cl_kernel kernel_permute3D_1_0_2;
    cl_kernel kernel_add_matmul_attention;
    cl_kernel kernel_softmax;
    cl_kernel kernel_matmul_attention;
    cl_kernel kernel_batch_matmul_mask;
    cl_kernel kernel_batch_matmul;

    cl_mem bufferAttentionMask;
};


#endif //MY_OPENCL_MULTIHEADATTENTION_H
