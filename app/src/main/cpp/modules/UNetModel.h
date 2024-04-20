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
#include "nn/UpSample.h"
#include "kernel/unit/LayerNormKernel.h"
#include "kernel/unit/LinearKernel.h"
#include "kernel/unit/UtilKernel.h"

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
    void concat_buffer(cl_mem input1, cl_mem input2, cl_mem output,
                         cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event);

    void initInputBlock0();
    void initInputBlock1();
    void initInputBlock2();
    void initInputBlock3();
    void initInputBlock4();
    void initInputBlock5();
    void initInputBlock6();
    void initInputBlock7();
    void initInputBlock8();
    void initInputBlock9();
    void initInputBlock10();
    void initInputBlock11();

    void initMiddleBlock();

    void initOutputBlock0();
    void initOutputBlock1();
    void initOutputBlock2();
    void initOutputBlock3();
    void initOutputBlock4();
    void initOutputBlock5();
    void initOutputBlock6();
    void initOutputBlock7();
    void initOutputBlock8();
    void initOutputBlock9();
    void initOutputBlock10();
    void initOutputBlock11();

    void initOut();

    cl_context context;
    cl_command_queue cmdQueue;
    cl_device_id deviceId;
    AAssetManager *assetManager;

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

    ResBlock *output_block_1_res_block;

    ResBlock *output_block_2_res_block;
    UpSample *output_block_2_up_sample;

    ResBlock *output_block_3_res_block;
    SpatialTransformer *output_block_3_spatial;

    ResBlock *output_block_4_res_block;
    SpatialTransformer *output_block_4_spatial;

    ResBlock *output_block_5_res_block;
    SpatialTransformer *output_block_5_spatial;
    UpSample *output_block_5_up_sample;

    ResBlock *output_block_6_res_block;
    SpatialTransformer *output_block_6_spatial;

    ResBlock *output_block_7_res_block;
    SpatialTransformer *output_block_7_spatial;

    ResBlock *output_block_8_res_block;
    SpatialTransformer *output_block_8_spatial;
    UpSample *output_block_8_up_sample;

    ResBlock *output_block_9_res_block;
    SpatialTransformer *output_block_9_spatial;

    ResBlock *output_block_10_res_block;
    SpatialTransformer *output_block_10_spatial;

    ResBlock *output_block_11_res_block;
    SpatialTransformer *output_block_11_spatial;

    GroupNorm *out_group_norm;
    Conv2D *out_conv2d;

    std::shared_ptr<LayerNormKernel> layerNormKernel;
    std::shared_ptr<LinearKernel> linearKernel;
    UtilKernel utilKernel;
};


#endif //MY_OPENCL_UNETMODEL_H
