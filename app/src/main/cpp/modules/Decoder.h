//
// Created by 구현우 on 2023/12/14.
//

#ifndef MY_OPENCL_DECODER_H
#define MY_OPENCL_DECODER_H

#include <android/asset_manager_jni.h>
#include <vector>
#include "nn/Conv2D.h"
#include "nn/ResBlock.h"

#define CL_TARGET_OPENCL_VERSION 200
#include "CL/opencl.h"

class Decoder {
public:
    Decoder(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
            AAssetManager *assetManager);

    ~Decoder();

    std::vector<float> decode(const std::vector<float> &x);

private:
    cl_context context;
    cl_command_queue cmdQueue;
    cl_device_id deviceId;

    Conv2D *post_quant_conv2d;

    Conv2D *in_conv2d;

    ResBlock *mid_res_block_1;
};


#endif //MY_OPENCL_DECODER_H
