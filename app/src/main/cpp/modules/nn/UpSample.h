//
// Created by 구현우 on 2023/12/11.
//

#ifndef MY_OPENCL_UPSAMPLE_H
#define MY_OPENCL_UPSAMPLE_H


#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <android/asset_manager_jni.h>

#include "Conv2D.h"

class UpSample {
public:
    UpSample(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
             AAssetManager *assetManager, const char *weight_name, const char *bias_name,
             int stride,
             int padding);

    ~UpSample();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_context context;
    cl_command_queue cmdQueue;

    cl_kernel kernel_up_sample;

    Conv2D *conv2d;

    size_t scale;
};


#endif //MY_OPENCL_UPSAMPLE_H
