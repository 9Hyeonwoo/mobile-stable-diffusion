//
// Created by 구현우 on 2023/12/10.
//

#ifndef MY_OPENCL_GEGLU_H
#define MY_OPENCL_GEGLU_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include "Linear.h"
#include <vector>
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/GEGLUKernel.h"
#include "../kernel/unit/UtilKernel.h"

class GEGLU {
public:
    GEGLU(
            cl_context context, cl_command_queue cmdQueue,
            size_t in_features, size_t out_features,
            const std::string &linear_weight_name, const std::string &linear_bias_name,
            std::shared_ptr<LinearKernel> linearKernel,
            std::shared_ptr<GEGLUKernel> gegluKernel,
            std::shared_ptr<UtilKernel> utilKernel
    );

    ~GEGLU();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

    std::vector<size_t> weightShape;
private:
    cl_command_queue cmdQueue;
    cl_context context;

    std::shared_ptr<GEGLUKernel> kernel;

    Linear *linear;
};


#endif //MY_OPENCL_GEGLU_H
