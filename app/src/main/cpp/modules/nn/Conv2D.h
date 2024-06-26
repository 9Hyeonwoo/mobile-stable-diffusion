//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_CONV2D_H
#define MY_OPENCL_CONV2D_H

#include <android/asset_manager_jni.h>
#include <vector>
#include <string>

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "../kernel/unit/ConvKernel.h"

class Conv2D {
public:
    Conv2D(cl_context context, cl_command_queue cmdQueue,
           size_t in_channel, size_t out_channel, size_t kernel_size, int stride, int padding,
           const std::string &weight_name, const std::string &bias_name,
           std::shared_ptr<ConvKernel> kernel);

    ~Conv2D();

    void init();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    std::vector<size_t> weightShape;
private:
    size_t getOutputSize(size_t inputSize);

    std::vector<size_t> biasShape;
    cl_context context;
    cl_command_queue cmdQueue;

    std::shared_ptr<ConvKernel> kernel;

    cl_mem bufferWeight;
    cl_mem bufferBias;

    int stride;
    int padding;

    const std::string weight_name;
    const std::string bias_name;
};


#endif //MY_OPENCL_CONV2D_H
