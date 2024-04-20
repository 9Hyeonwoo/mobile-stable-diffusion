//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_RESBLOCK_H
#define MY_OPENCL_RESBLOCK_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

#include <android/asset_manager_jni.h>
#include "GroupNorm.h"
#include "Conv2D.h"
#include "Linear.h"
#include "../kernel/unit/LinearKernel.h"

class ResBlock {
public:
    ResBlock(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
             AAssetManager *assetManager,
             size_t in_channels, size_t emb_channels, size_t out_channels,
             const std::string &in_group_norm_weight_name, const std::string &in_group_norm_bias_name,
             const std::string &in_conv2d_weight_name, const std::string &in_conv2d_bias_name,
             const std::string& embed_linear_weight_name, const std::string& embed_linear_bias_name,
             const std::string &out_group_norm_weight_name, const std::string &out_group_norm_bias_name,
             const std::string &out_conv2d_weight_name, const std::string &out_conv2d_bias_name,
             const std::string& skip_conv2d_weight_name, const std::string& skip_conv2d_bias_name,
             LinearKernel &linearKernel);

    ~ResBlock();

    void init();

    cl_int forward(cl_mem &input, cl_mem embed, cl_mem output,
                   cl_uint num_events_embed, const cl_event *event_wait_list_embed,
                   cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event);

private:
    cl_context context;
    cl_command_queue cmdQueue;

    size_t in_channels;
    size_t out_channels;

    cl_kernel kernel_silu;
    cl_kernel kernel_chunk_add;
    cl_kernel kernel_elem_add;

    GroupNorm *in_group_norm;
    Conv2D *in_conv2d;

    Linear *embed_linear;

    GroupNorm *out_group_norm;
    Conv2D *out_conv2d;

    Conv2D *skip_conv2d;

    static int cnt;
};


#endif //MY_OPENCL_RESBLOCK_H
