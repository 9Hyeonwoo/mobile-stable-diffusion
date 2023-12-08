//
// Created by 구현우 on 2023/12/08.
//

#ifndef MY_OPENCL_SPATIALTRANSFORMER_H
#define MY_OPENCL_SPATIALTRANSFORMER_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <android/asset_manager_jni.h>

#include "GroupNorm.h"
#include "Linear.h"

class SpatialTransformer {
public:
    SpatialTransformer(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                       AAssetManager *assetManager,
                       size_t channels,
                       const char *group_norm_weight_name, const char *group_norm_bias_name,
                       const char *in_linear_weight_name, const char *in_linear_bias_name);

    ~SpatialTransformer();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    size_t channels;
    cl_command_queue cmdQueue;
    cl_context context;

    GroupNorm *groupNorm;
    Linear *projInLinear;

    cl_kernel kernel_permute_0_2_1;
};


#endif //MY_OPENCL_SPATIALTRANSFORMER_H