//
// Created by 구현우 on 2023/12/14.
//

#ifndef MY_OPENCL_DECODER_H
#define MY_OPENCL_DECODER_H

#include <android/asset_manager_jni.h>
#include <vector>
#include "nn/Conv2D.h"
#include "nn/ResBlock.h"
#include "nn/AttnBlock.h"
#include "nn/UpSample.h"
#include "nn/GroupNorm.h"

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "kernel/unit/LinearKernel.h"

class Decoder {
public:
    Decoder(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
            AAssetManager *assetManager);

    ~Decoder();

    std::vector<float> decode(const std::vector<float> &x);

    void test(const std::vector<float> &x);

private:
    cl_context context;
    cl_command_queue cmdQueue;

    cl_kernel kernel_silu;

    Conv2D *post_quant_conv2d;

    Conv2D *in_conv2d;

    ResBlock *mid_res_block_1;

    AttnBlock *mid_attn_block;

    ResBlock *mid_res_block_2;

    ResBlock *up_3_res_blocks[3]{};
    UpSample *up_3_up_sample;

    ResBlock *up_2_res_blocks[3]{};
    UpSample *up_2_up_sample;

    ResBlock *up_1_res_blocks[3]{};
    UpSample *up_1_up_sample;

    ResBlock *up_0_res_blocks[3]{};

    GroupNorm *out_group_norm;
    Conv2D *out_conv2d;

    std::shared_ptr<LinearKernel> linearKernel;
};


#endif //MY_OPENCL_DECODER_H
