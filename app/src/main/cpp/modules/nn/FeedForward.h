//
// Created by 구현우 on 2023/12/10.
//

#ifndef MY_OPENCL_FEEDFORWARD_H
#define MY_OPENCL_FEEDFORWARD_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include "Linear.h"
#include "GEGLU.h"
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/GEGLUKernel.h"
#include "../kernel/unit/UtilKernel.h"

class FeedForward {
public:
    FeedForward(
            cl_context context, cl_command_queue cmdQueue,
            size_t dim,
            const std::string &geglu_linear_weight_name, const std::string &geglu_linear_bias_name,
            const std::string &net_linear_weight_name, const std::string &net_linear_bias_name,
            std::shared_ptr<LinearKernel> linearKernel,
            std::shared_ptr<GEGLUKernel> gegluKernel,
            std::shared_ptr<UtilKernel> utilKernel
    );

    ~FeedForward();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    cl_context context;
    cl_command_queue cmdQueue;

    GEGLU *geglu;
    Linear *netLinear;
};


#endif //MY_OPENCL_FEEDFORWARD_H
