//
// Created by 구현우 on 2023/12/01.
//

#ifndef MY_OPENCL_LINEAR_H
#define MY_OPENCL_LINEAR_H

#include "android/asset_manager_jni.h"

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <vector>

class Linear {
public:
    Linear(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
           AAssetManager *assetManager, const char *weight_name, const char *bias_name);

    ~Linear();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_mem bufferWeight;
    cl_mem bufferBias;
    std::vector<size_t> weightShape;
    std::vector<size_t> biasShape;
    cl_command_queue cmdQueue;

    cl_kernel kernel;
};


#endif //MY_OPENCL_LINEAR_H
