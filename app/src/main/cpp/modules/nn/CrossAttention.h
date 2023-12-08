//
// Created by 구현우 on 2023/12/08.
//

#ifndef MY_OPENCL_CROSSATTENTION_H
#define MY_OPENCL_CROSSATTENTION_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include <android/asset_manager_jni.h>
#include "Linear.h"

class CrossAttention {
public:
    CrossAttention(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                   AAssetManager *assetManager, size_t headSize, size_t headDim,
                   const char *q_linear_weight_name,
                   const char *k_linear_weight_name,
                   const char *v_linear_weight_name);

    ~CrossAttention();

    cl_int forward(cl_mem input, cl_mem condition, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_command_queue cmdQueue;
    cl_context context;
    size_t headSize;
    float scale;

    Linear *toQLinear;
    Linear *toKLinear;
    Linear *toVLinear;

    cl_kernel kernel_permute3D_1_0_2;
    cl_kernel kernel_einsum_bik_bjk_bij;
};


#endif //MY_OPENCL_CROSSATTENTION_H
