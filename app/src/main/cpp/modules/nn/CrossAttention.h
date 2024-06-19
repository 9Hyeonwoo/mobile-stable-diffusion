//
// Created by 구현우 on 2023/12/08.
//

#ifndef MY_OPENCL_CROSSATTENTION_H
#define MY_OPENCL_CROSSATTENTION_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "Linear.h"
#include "../kernel/unit/LinearKernel.h"
#include "../kernel/unit/UtilKernel.h"
#include "../kernel/unit/CrossAttentionKernel.h"

class CrossAttention {
public:
    CrossAttention(cl_context context, cl_command_queue cmdQueue,
                   size_t query_dim, size_t context_dim, size_t headSize, size_t headDim,
                   const std::string &q_linear_weight_name,
                   const std::string &k_linear_weight_name,
                   const std::string &v_linear_weight_name,
                   const std::string &out_linear_weight_name,
                   const std::string &out_linear_bias_name,
                   std::shared_ptr<LinearKernel> linearKernel,
                   std::shared_ptr<UtilKernel> utilKernel,
                   std::shared_ptr<CrossAttentionKernel> crossAttentionKernel
    );

    ~CrossAttention();

    cl_int forward(cl_mem input, cl_mem condition, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    cl_command_queue cmdQueue;
    cl_context context;
    size_t headSize;
    float scale;
    static int cnt;

    Linear *toQLinear;
    Linear *toKLinear;
    Linear *toVLinear;
    Linear *toOutLinear;

    std::shared_ptr<UtilKernel> utilKernel;
    std::shared_ptr<CrossAttentionKernel> crossAttentionKernel;
};


#endif //MY_OPENCL_CROSSATTENTION_H
