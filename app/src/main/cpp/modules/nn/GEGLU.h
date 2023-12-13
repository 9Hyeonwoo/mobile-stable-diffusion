//
// Created by 구현우 on 2023/12/10.
//

#ifndef MY_OPENCL_GEGLU_H
#define MY_OPENCL_GEGLU_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <android/asset_manager_jni.h>

#include "Linear.h"
#include <vector>

class GEGLU {
public:
    GEGLU(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
          AAssetManager *assetManager,
          size_t in_features, size_t out_features,
          const char *linear_weight_name, const char *linear_bias_name);

    ~GEGLU();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    std::vector<size_t> weightShape;
private:
    cl_command_queue cmdQueue;
    cl_context context;

    cl_kernel kernel_gelu_multiply;

    Linear *linear;
};


#endif //MY_OPENCL_GEGLU_H
