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

    void
    test(const std::vector<float> &x, long timestep, const std::vector<float> &condition);
private:
    std::vector<float> timestep_embedding(long timestep);

    cl_context context;
    cl_command_queue cmdQueue;

    Linear *time_embed_0;
    Linear *time_embed_2;

    Conv2D *input_block_0_conv2d;

    ResBlock *input_block_1_res_block;
    SpatialTransformer *input_block_1_spatial;

    ResBlock *input_block_2_res_block;
    SpatialTransformer *input_block_2_spatial;

    Conv2D *input_block_3_conv2d;

    ResBlock *input_block_4_res_block;
    SpatialTransformer *input_block_4_spatial;

    ResBlock *input_block_5_res_block;
    SpatialTransformer *input_block_5_spatial;

    Conv2D *input_block_6_conv2d;

    ResBlock *input_block_7_res_block;
    SpatialTransformer *input_block_7_spatial;

    ResBlock *input_block_8_res_block;
    SpatialTransformer *input_block_8_spatial;

    Conv2D *input_block_9_conv2d;

    ResBlock *input_block_10_res_block;

    ResBlock *input_block_11_res_block;

    ResBlock *middle_block_0_res_block;

    SpatialTransformer *middle_block_1_spatial;

    ResBlock *middle_block_2_res_block;

    ResBlock *output_block_0_res_block;

    cl_kernel kernel_silu;
};


#endif //MY_OPENCL_UNETMODEL_H
