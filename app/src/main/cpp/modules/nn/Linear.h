//
// Created by 구현우 on 2023/12/01.
//

#ifndef MY_OPENCL_LINEAR_H
#define MY_OPENCL_LINEAR_H

#include "android/asset_manager_jni.h"

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <vector>
#include <string>
#include "../kernel/unit/LinearKernel.h"

class Linear {
public:
    Linear(cl_context context, cl_command_queue cmdQueue,
           size_t in_features, size_t out_features,
           const std::string &weight_name, const std::string &bias_name,
           LinearKernel &kernel);

    ~Linear();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

    std::vector<size_t> weightShape;
private:
    cl_mem bufferWeight;
    /* bufferBias is nullable */
    cl_mem bufferBias;

    cl_command_queue cmdQueue;
    cl_context context;

    const std::string weight_name;
    const std::string bias_name;

    LinearKernel kernel;
};


#endif //MY_OPENCL_LINEAR_H
