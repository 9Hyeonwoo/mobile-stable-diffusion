//
// Created by 구현우 on 2023/12/14.
//

#ifndef MY_OPENCL_ATTNBLOCK_H
#define MY_OPENCL_ATTNBLOCK_H
#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include <android/asset_manager_jni.h>

#include "GroupNorm.h"
#include "Conv2D.h"
#include "../kernel/unit/ConvKernel.h"
#include "../kernel/unit/UtilKernel.h"
#include "../kernel/unit/GroupNormKernel.h"

class AttnBlock {
public:
    AttnBlock(
            cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
            AAssetManager *assetManager,
            size_t in_channels,
            const std::string &group_norm_weight, const std::string &group_norm_bias,
            const std::string &q_conv2d_weight_name, const std::string &q_conv2d_bias_name,
            const std::string &k_conv2d_weight_name, const std::string &k_conv2d_bias_name,
            const std::string &v_conv2d_weight_name, const std::string &v_conv2d_bias_name,
            const std::string &out_conv2d_weight_name, const std::string &out_conv2d_bias_name,
            std::shared_ptr<ConvKernel> convKernel,
            std::shared_ptr<UtilKernel> utilKernel,
            std::shared_ptr<GroupNormKernel> groupNormKernel
    );

    ~AttnBlock();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

    void init();

private:
    cl_command_queue cmdQueue;
    cl_context context;
    size_t in_channels;

    std::shared_ptr<UtilKernel> utilKernel;

    GroupNorm *groupNorm;

    Conv2D *to_q_conv2d;
    Conv2D *to_k_conv2d;
    Conv2D *to_v_conv2d;
    Conv2D *out_conv2d;
};


#endif //MY_OPENCL_ATTNBLOCK_H
