//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_UNETMODEL_H
#define MY_OPENCL_UNETMODEL_H

#include <vector>
#include <android/asset_manager_jni.h>
#include "nn/Linear.h"
#include "nn/Conv2D.h"
#include "nn/ResBlock.h"
#include "nn/SpatialTransformer.h"

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"

class UNetModel {
public:
    UNetModel(AAssetManager *assetManager, cl_context context, cl_command_queue cmdQueue,
              cl_device_id deviceId);

    ~UNetModel();

    std::vector<float>
    forward(const std::vector<float> &x, long timestep, const std::vector<float> &condition);

private:
    std::vector<float> timestep_embedding(long timestep);

    cl_context context;
    cl_command_queue cmdQueue;

    Linear *time_embed_0;
    Linear *time_embed_2;

    Conv2D *input_block_0_conv2d;

    ResBlock *input_block_1_res_block;

    SpatialTransformer *input_block_1_spatial;

    cl_kernel kernel_silu;
};


#endif //MY_OPENCL_UNETMODEL_H
