//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_GROUPNORM_H
#define MY_OPENCL_GROUPNORM_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <string>
#include "../kernel/unit/GroupNormKernel.h"

class GroupNorm {
public:
    GroupNorm(
            cl_context context, cl_command_queue cmdQueue,
            size_t num_groups, size_t num_channels, float eps,
            const std::string &weight_name, const std::string &bias_name,
            std::shared_ptr<GroupNormKernel> kernel
    );

    ~GroupNorm();

    void init();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_mem bufferWeight;
    cl_mem bufferBias;
    size_t weightSize;
    size_t biasSize;
    size_t num_groups;
    size_t num_channels;
    float eps;

    cl_command_queue cmdQueue;
    cl_context context;

    std::shared_ptr<GroupNormKernel> kernel;

    const std::string weight_name;
    const std::string bias_name;
};


#endif //MY_OPENCL_GROUPNORM_H
