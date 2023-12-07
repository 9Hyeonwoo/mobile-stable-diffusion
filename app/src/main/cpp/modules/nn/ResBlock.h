//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_RESBLOCK_H
#define MY_OPENCL_RESBLOCK_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <android/asset_manager_jni.h>
#include "GroupNorm.h"
#include "Conv2D.h"

class ResBlock {
public:
    ResBlock(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
             AAssetManager *assetManager,
             const char *in_group_norm_weight_name, const char *in_group_norm_bias_name,
             const char *in_conv2d_weight_name, const char *in_conv2d_bias_name);

    ~ResBlock();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_context context;
    cl_command_queue cmdQueue;
    AAssetManager *assetManager;

    cl_kernel kernel_silu;

    GroupNorm *in_group_norm;
    Conv2D *in_conv2d;
};


#endif //MY_OPENCL_RESBLOCK_H
