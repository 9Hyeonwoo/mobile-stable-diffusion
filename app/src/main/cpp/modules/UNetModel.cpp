//
// Created by 구현우 on 2023/12/07.
//

#include "UNetModel.h"

#include "util.h"
#include <android/log.h>

#define LOG_TAG "UNET_MODEL"
#define MODEL_CHANNELS 320
#define TIME_EMBED_DIM (4 * MODEL_CHANNELS)
#define CONTEXT_DIM 1024
#define NUM_HEAD_CHANNELS 64

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

UNetModel::UNetModel(
        AAssetManager *assetManager,
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId
) : context(context), cmdQueue(cmdQueue), deviceId(deviceId), assetManager(assetManager) {
    cl_int err;
    time_embed_0 = new Linear(context, cmdQueue, deviceId, assetManager,
                              320, 1280,
                              "unet/time_embed/time_embed_0_weight.npy",
                              "unet/time_embed/time_embed_0_bias.npy");
    time_embed_0->init();

    time_embed_2 = new Linear(context, cmdQueue, deviceId, assetManager,
                              1280, 1280,
                              "unet/time_embed/time_embed_2_weight.npy",
                              "unet/time_embed/time_embed_2_bias.npy");
    time_embed_2->init();

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR(err);

    clReleaseProgram(program);
}

void UNetModel::initInputBlock0() {
    input_block_0_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      4, 320, 3, 1, 1,
                                      "unet/input_block/0/input_block_0_conv2d_weight.npy",
                                      "unet/input_block/0/input_block_0_conv2d_bias.npy");
}

void UNetModel::initInputBlock1() {
    input_block_1_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, TIME_EMBED_DIM, 320,
                                           "unet/input_block/1/input_block_1_res_block_in_group_norm_weight.npy",
                                           "unet/input_block/1/input_block_1_res_block_in_group_norm_bias.npy",
                                           "unet/input_block/1/input_block_1_res_block_in_conv2d_weight.npy",
                                           "unet/input_block/1/input_block_1_res_block_in_conv2d_bias.npy",
                                           "unet/input_block/1/input_block_1_res_block_embed_linear_weight.npy",
                                           "unet/input_block/1/input_block_1_res_block_embed_linear_bias.npy",
                                           "unet/input_block/1/input_block_1_res_block_out_group_norm_weight.npy",
                                           "unet/input_block/1/input_block_1_res_block_out_group_norm_bias.npy",
                                           "unet/input_block/1/input_block_1_res_block_out_conv2d_weight.npy",
                                           "unet/input_block/1/input_block_1_res_block_out_conv2d_bias.npy",
                                           nullptr, nullptr);

    input_block_1_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   320, CONTEXT_DIM, 5, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/1/input_block_1_spatial_group_norm_weight.npy",
                                                   "unet/input_block/1/input_block_1_spatial_group_norm_bias.npy",
                                                   "unet/input_block/1/input_block_1_spatial_in_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_spatial_in_linear_bias.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_1_weight.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_1_bias.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_2_weight.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_2_bias.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_3_weight.npy",
                                                   "unet/input_block/1/input_block_1_basic_layer_norm_3_bias.npy",
                                                   "unet/input_block/1/input_block_1_cross_1_q_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_1_k_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_1_v_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_1_out_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_1_out_linear_bias.npy",
                                                   "unet/input_block/1/input_block_1_cross_2_q_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_2_k_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_2_v_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_2_out_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_cross_2_out_linear_bias.npy",
                                                   "unet/input_block/1/input_block_1_ff_geglu_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_ff_geglu_linear_bias.npy",
                                                   "unet/input_block/1/input_block_1_ff_net_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_ff_net_linear_bias.npy",
                                                   "unet/input_block/1/input_block_1_spatial_out_linear_weight.npy",
                                                   "unet/input_block/1/input_block_1_spatial_out_linear_bias.npy");
}

void UNetModel::initInputBlock2() {
    input_block_2_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, TIME_EMBED_DIM, 320,
                                           "unet/input_block/2/input_blocks_2_0_in_layers_0_weight.npy",
                                           "unet/input_block/2/input_blocks_2_0_in_layers_0_bias.npy",
                                           "unet/input_block/2/input_blocks_2_0_in_layers_2_weight.npy",
                                           "unet/input_block/2/input_blocks_2_0_in_layers_2_bias.npy",
                                           "unet/input_block/2/input_blocks_2_0_emb_layers_1_weight.npy",
                                           "unet/input_block/2/input_blocks_2_0_emb_layers_1_bias.npy",
                                           "unet/input_block/2/input_blocks_2_0_out_layers_0_weight.npy",
                                           "unet/input_block/2/input_blocks_2_0_out_layers_0_bias.npy",
                                           "unet/input_block/2/input_blocks_2_0_out_layers_3_weight.npy",
                                           "unet/input_block/2/input_blocks_2_0_out_layers_3_bias.npy",
                                           nullptr, nullptr);

    input_block_2_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   320, CONTEXT_DIM, 5, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/2/input_blocks_2_1_norm_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_norm_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_proj_in_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_proj_in_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm1_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm1_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm2_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm2_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm3_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_norm3_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                   "unet/input_block/2/input_blocks_2_1_proj_out_weight.npy",
                                                   "unet/input_block/2/input_blocks_2_1_proj_out_bias.npy");

}

void UNetModel::initInputBlock3() {
    input_block_3_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      320, 320, 3, 2, 1,
                                      "unet/input_block/3/input_blocks_3_0_op_weight.npy",
                                      "unet/input_block/3/input_blocks_3_0_op_bias.npy");
}

void UNetModel::initInputBlock4() {
    input_block_4_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, TIME_EMBED_DIM, 640,
                                           "unet/input_block/4/input_blocks_4_0_in_layers_0_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_in_layers_0_bias.npy",
                                           "unet/input_block/4/input_blocks_4_0_in_layers_2_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_in_layers_2_bias.npy",
                                           "unet/input_block/4/input_blocks_4_0_emb_layers_1_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_emb_layers_1_bias.npy",
                                           "unet/input_block/4/input_blocks_4_0_out_layers_0_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_out_layers_0_bias.npy",
                                           "unet/input_block/4/input_blocks_4_0_out_layers_3_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_out_layers_3_bias.npy",
                                           "unet/input_block/4/input_blocks_4_0_skip_connection_weight.npy",
                                           "unet/input_block/4/input_blocks_4_0_skip_connection_bias.npy");

    input_block_4_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   640, CONTEXT_DIM, 10, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/4/input_blocks_4_1_norm_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_norm_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_proj_in_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_proj_in_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm1_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm1_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm2_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm2_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm3_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_norm3_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                   "unet/input_block/4/input_blocks_4_1_proj_out_weight.npy",
                                                   "unet/input_block/4/input_blocks_4_1_proj_out_bias.npy");
}

void UNetModel::initInputBlock5() {
    input_block_5_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           640, TIME_EMBED_DIM, 640,
                                           "unet/input_block/5/input_blocks_5_0_in_layers_0_weight.npy",
                                           "unet/input_block/5/input_blocks_5_0_in_layers_0_bias.npy",
                                           "unet/input_block/5/input_blocks_5_0_in_layers_2_weight.npy",
                                           "unet/input_block/5/input_blocks_5_0_in_layers_2_bias.npy",
                                           "unet/input_block/5/input_blocks_5_0_emb_layers_1_weight.npy",
                                           "unet/input_block/5/input_blocks_5_0_emb_layers_1_bias.npy",
                                           "unet/input_block/5/input_blocks_5_0_out_layers_0_weight.npy",
                                           "unet/input_block/5/input_blocks_5_0_out_layers_0_bias.npy",
                                           "unet/input_block/5/input_blocks_5_0_out_layers_3_weight.npy",
                                           "unet/input_block/5/input_blocks_5_0_out_layers_3_bias.npy",
                                           nullptr, nullptr);

    input_block_5_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   640, CONTEXT_DIM, 10, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/5/input_blocks_5_1_norm_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_norm_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_proj_in_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_proj_in_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm1_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm1_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm2_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm2_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm3_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_norm3_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                   "unet/input_block/5/input_blocks_5_1_proj_out_weight.npy",
                                                   "unet/input_block/5/input_blocks_5_1_proj_out_bias.npy");

}

void UNetModel::initInputBlock6() {
    input_block_6_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      640, 640, 3, 2, 1,
                                      "unet/input_block/6/input_blocks_6_0_op_weight.npy",
                                      "unet/input_block/6/input_blocks_6_0_op_bias.npy");
}

void UNetModel::initInputBlock7() {
    input_block_7_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           640, TIME_EMBED_DIM, 1280,
                                           "unet/input_block/7/input_blocks_7_0_in_layers_0_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_in_layers_0_bias.npy",
                                           "unet/input_block/7/input_blocks_7_0_in_layers_2_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_in_layers_2_bias.npy",
                                           "unet/input_block/7/input_blocks_7_0_emb_layers_1_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_emb_layers_1_bias.npy",
                                           "unet/input_block/7/input_blocks_7_0_out_layers_0_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_out_layers_0_bias.npy",
                                           "unet/input_block/7/input_blocks_7_0_out_layers_3_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_out_layers_3_bias.npy",
                                           "unet/input_block/7/input_blocks_7_0_skip_connection_weight.npy",
                                           "unet/input_block/7/input_blocks_7_0_skip_connection_bias.npy");

    input_block_7_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/7/input_blocks_7_1_norm_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_norm_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_proj_in_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_proj_in_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm1_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm1_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm2_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm2_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm3_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_norm3_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                   "unet/input_block/7/input_blocks_7_1_proj_out_weight.npy",
                                                   "unet/input_block/7/input_blocks_7_1_proj_out_bias.npy");
}

void UNetModel::initInputBlock8() {
    input_block_8_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           1280, TIME_EMBED_DIM, 1280,
                                           "unet/input_block/8/input_blocks_8_0_in_layers_0_weight.npy",
                                           "unet/input_block/8/input_blocks_8_0_in_layers_0_bias.npy",
                                           "unet/input_block/8/input_blocks_8_0_in_layers_2_weight.npy",
                                           "unet/input_block/8/input_blocks_8_0_in_layers_2_bias.npy",
                                           "unet/input_block/8/input_blocks_8_0_emb_layers_1_weight.npy",
                                           "unet/input_block/8/input_blocks_8_0_emb_layers_1_bias.npy",
                                           "unet/input_block/8/input_blocks_8_0_out_layers_0_weight.npy",
                                           "unet/input_block/8/input_blocks_8_0_out_layers_0_bias.npy",
                                           "unet/input_block/8/input_blocks_8_0_out_layers_3_weight.npy",
                                           "unet/input_block/8/input_blocks_8_0_out_layers_3_bias.npy",
                                           nullptr, nullptr);

    input_block_8_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                   1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                   "unet/input_block/8/input_blocks_8_1_norm_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_norm_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_proj_in_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_proj_in_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm1_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm1_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm2_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm2_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm3_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_norm3_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                   "unet/input_block/8/input_blocks_8_1_proj_out_weight.npy",
                                                   "unet/input_block/8/input_blocks_8_1_proj_out_bias.npy");
}

void UNetModel::initInputBlock9() {
    input_block_9_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      1280, 1280, 3, 2, 1,
                                      "unet/input_block/9/input_blocks_9_0_op_weight.npy",
                                      "unet/input_block/9/input_blocks_9_0_op_bias.npy");
}

void UNetModel::initInputBlock10() {
    input_block_10_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, TIME_EMBED_DIM, 1280,
                                            "unet/input_block/10/input_blocks_10_0_in_layers_0_weight.npy",
                                            "unet/input_block/10/input_blocks_10_0_in_layers_0_bias.npy",
                                            "unet/input_block/10/input_blocks_10_0_in_layers_2_weight.npy",
                                            "unet/input_block/10/input_blocks_10_0_in_layers_2_bias.npy",
                                            "unet/input_block/10/input_blocks_10_0_emb_layers_1_weight.npy",
                                            "unet/input_block/10/input_blocks_10_0_emb_layers_1_bias.npy",
                                            "unet/input_block/10/input_blocks_10_0_out_layers_0_weight.npy",
                                            "unet/input_block/10/input_blocks_10_0_out_layers_0_bias.npy",
                                            "unet/input_block/10/input_blocks_10_0_out_layers_3_weight.npy",
                                            "unet/input_block/10/input_blocks_10_0_out_layers_3_bias.npy",
                                            nullptr, nullptr);
}

void UNetModel::initInputBlock11() {
    input_block_11_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, TIME_EMBED_DIM, 1280,
                                            "unet/input_block/11/input_blocks_11_0_in_layers_0_weight.npy",
                                            "unet/input_block/11/input_blocks_11_0_in_layers_0_bias.npy",
                                            "unet/input_block/11/input_blocks_11_0_in_layers_2_weight.npy",
                                            "unet/input_block/11/input_blocks_11_0_in_layers_2_bias.npy",
                                            "unet/input_block/11/input_blocks_11_0_emb_layers_1_weight.npy",
                                            "unet/input_block/11/input_blocks_11_0_emb_layers_1_bias.npy",
                                            "unet/input_block/11/input_blocks_11_0_out_layers_0_weight.npy",
                                            "unet/input_block/11/input_blocks_11_0_out_layers_0_bias.npy",
                                            "unet/input_block/11/input_blocks_11_0_out_layers_3_weight.npy",
                                            "unet/input_block/11/input_blocks_11_0_out_layers_3_bias.npy",
                                            nullptr, nullptr);
}


void UNetModel::initMiddleBlock() {
    middle_block_0_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, TIME_EMBED_DIM, 1280,
                                            "unet/middle_block/0/middle_block_0_in_layers_0_weight.npy",
                                            "unet/middle_block/0/middle_block_0_in_layers_0_bias.npy",
                                            "unet/middle_block/0/middle_block_0_in_layers_2_weight.npy",
                                            "unet/middle_block/0/middle_block_0_in_layers_2_bias.npy",
                                            "unet/middle_block/0/middle_block_0_emb_layers_1_weight.npy",
                                            "unet/middle_block/0/middle_block_0_emb_layers_1_bias.npy",
                                            "unet/middle_block/0/middle_block_0_out_layers_0_weight.npy",
                                            "unet/middle_block/0/middle_block_0_out_layers_0_bias.npy",
                                            "unet/middle_block/0/middle_block_0_out_layers_3_weight.npy",
                                            "unet/middle_block/0/middle_block_0_out_layers_3_bias.npy",
                                            nullptr, nullptr);

    middle_block_1_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                    "unet/middle_block/1/middle_block_1_norm_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_norm_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_proj_in_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_proj_in_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/middle_block/1/middle_block_1_proj_out_weight.npy",
                                                    "unet/middle_block/1/middle_block_1_proj_out_bias.npy");

    middle_block_2_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, TIME_EMBED_DIM, 1280,
                                            "unet/middle_block/2/middle_block_2_in_layers_0_weight.npy",
                                            "unet/middle_block/2/middle_block_2_in_layers_0_bias.npy",
                                            "unet/middle_block/2/middle_block_2_in_layers_2_weight.npy",
                                            "unet/middle_block/2/middle_block_2_in_layers_2_bias.npy",
                                            "unet/middle_block/2/middle_block_2_emb_layers_1_weight.npy",
                                            "unet/middle_block/2/middle_block_2_emb_layers_1_bias.npy",
                                            "unet/middle_block/2/middle_block_2_out_layers_0_weight.npy",
                                            "unet/middle_block/2/middle_block_2_out_layers_0_bias.npy",
                                            "unet/middle_block/2/middle_block_2_out_layers_3_weight.npy",
                                            "unet/middle_block/2/middle_block_2_out_layers_3_bias.npy",
                                            nullptr, nullptr);
}

void UNetModel::initOutputBlock0() {
    output_block_0_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/0/output_blocks_0_0_in_layers_0_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_in_layers_0_bias.npy",
                                            "unet/output_block/0/output_blocks_0_0_in_layers_2_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_in_layers_2_bias.npy",
                                            "unet/output_block/0/output_blocks_0_0_emb_layers_1_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_emb_layers_1_bias.npy",
                                            "unet/output_block/0/output_blocks_0_0_out_layers_0_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_out_layers_0_bias.npy",
                                            "unet/output_block/0/output_blocks_0_0_out_layers_3_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_out_layers_3_bias.npy",
                                            "unet/output_block/0/output_blocks_0_0_skip_connection_weight.npy",
                                            "unet/output_block/0/output_blocks_0_0_skip_connection_bias.npy");
}


void UNetModel::initOutputBlock1() {
    output_block_1_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/1/output_blocks_1_0_in_layers_0_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_in_layers_0_bias.npy",
                                            "unet/output_block/1/output_blocks_1_0_in_layers_2_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_in_layers_2_bias.npy",
                                            "unet/output_block/1/output_blocks_1_0_emb_layers_1_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_emb_layers_1_bias.npy",
                                            "unet/output_block/1/output_blocks_1_0_out_layers_0_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_out_layers_0_bias.npy",
                                            "unet/output_block/1/output_blocks_1_0_out_layers_3_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_out_layers_3_bias.npy",
                                            "unet/output_block/1/output_blocks_1_0_skip_connection_weight.npy",
                                            "unet/output_block/1/output_blocks_1_0_skip_connection_bias.npy");
}

void UNetModel::initOutputBlock2() {
    output_block_2_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/2/output_blocks_2_0_in_layers_0_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_in_layers_0_bias.npy",
                                            "unet/output_block/2/output_blocks_2_0_in_layers_2_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_in_layers_2_bias.npy",
                                            "unet/output_block/2/output_blocks_2_0_emb_layers_1_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_emb_layers_1_bias.npy",
                                            "unet/output_block/2/output_blocks_2_0_out_layers_0_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_out_layers_0_bias.npy",
                                            "unet/output_block/2/output_blocks_2_0_out_layers_3_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_out_layers_3_bias.npy",
                                            "unet/output_block/2/output_blocks_2_0_skip_connection_weight.npy",
                                            "unet/output_block/2/output_blocks_2_0_skip_connection_bias.npy");

    output_block_2_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                            1280, 1280, 3, 1, 1,
                                            "unet/output_block/2/output_blocks_2_1_conv_weight.npy",
                                            "unet/output_block/2/output_blocks_2_1_conv_bias.npy");
}

void UNetModel::initOutputBlock3() {
    output_block_3_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/3/output_blocks_3_0_in_layers_0_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_in_layers_0_bias.npy",
                                            "unet/output_block/3/output_blocks_3_0_in_layers_2_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_in_layers_2_bias.npy",
                                            "unet/output_block/3/output_blocks_3_0_emb_layers_1_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_emb_layers_1_bias.npy",
                                            "unet/output_block/3/output_blocks_3_0_out_layers_0_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_out_layers_0_bias.npy",
                                            "unet/output_block/3/output_blocks_3_0_out_layers_3_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_out_layers_3_bias.npy",
                                            "unet/output_block/3/output_blocks_3_0_skip_connection_weight.npy",
                                            "unet/output_block/3/output_blocks_3_0_skip_connection_bias.npy");

    output_block_3_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/3/output_blocks_3_1_norm_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_norm_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_proj_in_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_proj_in_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/3/output_blocks_3_1_proj_out_weight.npy",
                                                    "unet/output_block/3/output_blocks_3_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock4() {
    output_block_4_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/4/output_blocks_4_0_in_layers_0_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_in_layers_0_bias.npy",
                                            "unet/output_block/4/output_blocks_4_0_in_layers_2_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_in_layers_2_bias.npy",
                                            "unet/output_block/4/output_blocks_4_0_emb_layers_1_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_emb_layers_1_bias.npy",
                                            "unet/output_block/4/output_blocks_4_0_out_layers_0_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_out_layers_0_bias.npy",
                                            "unet/output_block/4/output_blocks_4_0_out_layers_3_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_out_layers_3_bias.npy",
                                            "unet/output_block/4/output_blocks_4_0_skip_connection_weight.npy",
                                            "unet/output_block/4/output_blocks_4_0_skip_connection_bias.npy");

    output_block_4_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/4/output_blocks_4_1_norm_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_norm_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_proj_in_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_proj_in_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/4/output_blocks_4_1_proj_out_weight.npy",
                                                    "unet/output_block/4/output_blocks_4_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock5() {
    output_block_5_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1920, TIME_EMBED_DIM, 1280,
                                            "unet/output_block/5/output_blocks_5_0_in_layers_0_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_in_layers_0_bias.npy",
                                            "unet/output_block/5/output_blocks_5_0_in_layers_2_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_in_layers_2_bias.npy",
                                            "unet/output_block/5/output_blocks_5_0_emb_layers_1_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_emb_layers_1_bias.npy",
                                            "unet/output_block/5/output_blocks_5_0_out_layers_0_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_out_layers_0_bias.npy",
                                            "unet/output_block/5/output_blocks_5_0_out_layers_3_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_out_layers_3_bias.npy",
                                            "unet/output_block/5/output_blocks_5_0_skip_connection_weight.npy",
                                            "unet/output_block/5/output_blocks_5_0_skip_connection_bias.npy");

    output_block_5_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    1280, CONTEXT_DIM, 20, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/5/output_blocks_5_1_norm_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_norm_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_proj_in_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_proj_in_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/5/output_blocks_5_1_proj_out_weight.npy",
                                                    "unet/output_block/5/output_blocks_5_1_proj_out_bias.npy");

    output_block_5_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                            1280, 1280, 3, 1, 1,
                                            "unet/output_block/5/output_blocks_5_2_conv_weight.npy",
                                            "unet/output_block/5/output_blocks_5_2_conv_bias.npy");
}

void UNetModel::initOutputBlock6() {
    output_block_6_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1920, TIME_EMBED_DIM, 640,
                                            "unet/output_block/6/output_blocks_6_0_in_layers_0_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_in_layers_0_bias.npy",
                                            "unet/output_block/6/output_blocks_6_0_in_layers_2_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_in_layers_2_bias.npy",
                                            "unet/output_block/6/output_blocks_6_0_emb_layers_1_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_emb_layers_1_bias.npy",
                                            "unet/output_block/6/output_blocks_6_0_out_layers_0_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_out_layers_0_bias.npy",
                                            "unet/output_block/6/output_blocks_6_0_out_layers_3_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_out_layers_3_bias.npy",
                                            "unet/output_block/6/output_blocks_6_0_skip_connection_weight.npy",
                                            "unet/output_block/6/output_blocks_6_0_skip_connection_bias.npy");

    output_block_6_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    640, CONTEXT_DIM, 10, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/6/output_blocks_6_1_norm_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_norm_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_proj_in_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_proj_in_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/6/output_blocks_6_1_proj_out_weight.npy",
                                                    "unet/output_block/6/output_blocks_6_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock7() {
    output_block_7_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, TIME_EMBED_DIM, 640,
                                            "unet/output_block/7/output_blocks_7_0_in_layers_0_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_in_layers_0_bias.npy",
                                            "unet/output_block/7/output_blocks_7_0_in_layers_2_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_in_layers_2_bias.npy",
                                            "unet/output_block/7/output_blocks_7_0_emb_layers_1_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_emb_layers_1_bias.npy",
                                            "unet/output_block/7/output_blocks_7_0_out_layers_0_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_out_layers_0_bias.npy",
                                            "unet/output_block/7/output_blocks_7_0_out_layers_3_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_out_layers_3_bias.npy",
                                            "unet/output_block/7/output_blocks_7_0_skip_connection_weight.npy",
                                            "unet/output_block/7/output_blocks_7_0_skip_connection_bias.npy");

    output_block_7_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    640, CONTEXT_DIM, 10, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/7/output_blocks_7_1_norm_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_norm_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_proj_in_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_proj_in_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/7/output_blocks_7_1_proj_out_weight.npy",
                                                    "unet/output_block/7/output_blocks_7_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock8() {
    output_block_8_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            960, TIME_EMBED_DIM, 640,
                                            "unet/output_block/8/output_blocks_8_0_in_layers_0_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_in_layers_0_bias.npy",
                                            "unet/output_block/8/output_blocks_8_0_in_layers_2_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_in_layers_2_bias.npy",
                                            "unet/output_block/8/output_blocks_8_0_emb_layers_1_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_emb_layers_1_bias.npy",
                                            "unet/output_block/8/output_blocks_8_0_out_layers_0_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_out_layers_0_bias.npy",
                                            "unet/output_block/8/output_blocks_8_0_out_layers_3_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_out_layers_3_bias.npy",
                                            "unet/output_block/8/output_blocks_8_0_skip_connection_weight.npy",
                                            "unet/output_block/8/output_blocks_8_0_skip_connection_bias.npy");

    output_block_8_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    640, CONTEXT_DIM, 10, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/8/output_blocks_8_1_norm_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_norm_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_proj_in_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_proj_in_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/8/output_blocks_8_1_proj_out_weight.npy",
                                                    "unet/output_block/8/output_blocks_8_1_proj_out_bias.npy");

    output_block_8_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                            640, 640, 3, 1, 1,
                                            "unet/output_block/8/output_blocks_8_2_conv_weight.npy",
                                            "unet/output_block/8/output_blocks_8_2_conv_bias.npy");
}

void UNetModel::initOutputBlock9() {
    output_block_9_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            960, TIME_EMBED_DIM, 320,
                                            "unet/output_block/9/output_blocks_9_0_in_layers_0_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_in_layers_0_bias.npy",
                                            "unet/output_block/9/output_blocks_9_0_in_layers_2_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_in_layers_2_bias.npy",
                                            "unet/output_block/9/output_blocks_9_0_emb_layers_1_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_emb_layers_1_bias.npy",
                                            "unet/output_block/9/output_blocks_9_0_out_layers_0_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_out_layers_0_bias.npy",
                                            "unet/output_block/9/output_blocks_9_0_out_layers_3_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_out_layers_3_bias.npy",
                                            "unet/output_block/9/output_blocks_9_0_skip_connection_weight.npy",
                                            "unet/output_block/9/output_blocks_9_0_skip_connection_bias.npy");

    output_block_9_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                    320, CONTEXT_DIM, 5, NUM_HEAD_CHANNELS,
                                                    "unet/output_block/9/output_blocks_9_1_norm_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_norm_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_proj_in_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_proj_in_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm1_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm1_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm2_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm2_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm3_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_norm3_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                    "unet/output_block/9/output_blocks_9_1_proj_out_weight.npy",
                                                    "unet/output_block/9/output_blocks_9_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock10() {
    output_block_10_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                             640, TIME_EMBED_DIM, 320,
                                             "unet/output_block/10/output_blocks_10_0_in_layers_0_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_in_layers_0_bias.npy",
                                             "unet/output_block/10/output_blocks_10_0_in_layers_2_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_in_layers_2_bias.npy",
                                             "unet/output_block/10/output_blocks_10_0_emb_layers_1_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_emb_layers_1_bias.npy",
                                             "unet/output_block/10/output_blocks_10_0_out_layers_0_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_out_layers_0_bias.npy",
                                             "unet/output_block/10/output_blocks_10_0_out_layers_3_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_out_layers_3_bias.npy",
                                             "unet/output_block/10/output_blocks_10_0_skip_connection_weight.npy",
                                             "unet/output_block/10/output_blocks_10_0_skip_connection_bias.npy");

    output_block_10_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                     320, CONTEXT_DIM, 5, NUM_HEAD_CHANNELS,
                                                     "unet/output_block/10/output_blocks_10_1_norm_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_norm_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_proj_in_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_proj_in_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm1_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm1_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm2_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm2_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm3_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_norm3_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                     "unet/output_block/10/output_blocks_10_1_proj_out_weight.npy",
                                                     "unet/output_block/10/output_blocks_10_1_proj_out_bias.npy");
}

void UNetModel::initOutputBlock11() {
    output_block_11_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                             640, TIME_EMBED_DIM, 320,
                                             "unet/output_block/11/output_blocks_11_0_in_layers_0_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_in_layers_0_bias.npy",
                                             "unet/output_block/11/output_blocks_11_0_in_layers_2_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_in_layers_2_bias.npy",
                                             "unet/output_block/11/output_blocks_11_0_emb_layers_1_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_emb_layers_1_bias.npy",
                                             "unet/output_block/11/output_blocks_11_0_out_layers_0_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_out_layers_0_bias.npy",
                                             "unet/output_block/11/output_blocks_11_0_out_layers_3_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_out_layers_3_bias.npy",
                                             "unet/output_block/11/output_blocks_11_0_skip_connection_weight.npy",
                                             "unet/output_block/11/output_blocks_11_0_skip_connection_bias.npy");

    output_block_11_spatial = new SpatialTransformer(context, cmdQueue, deviceId, assetManager,
                                                     320, CONTEXT_DIM, 5, NUM_HEAD_CHANNELS,
                                                     "unet/output_block/11/output_blocks_11_1_norm_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_norm_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_proj_in_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_proj_in_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm1_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm1_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm2_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm2_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm3_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_norm3_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn1_to_q_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn1_to_k_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn1_to_v_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn1_to_out_0_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn1_to_out_0_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn2_to_q_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn2_to_k_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn2_to_v_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn2_to_out_0_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_attn2_to_out_0_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_ff_net_0_proj_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_ff_net_0_proj_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_ff_net_2_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_transformer_blocks_0_ff_net_2_bias.npy",
                                                     "unet/output_block/11/output_blocks_11_1_proj_out_weight.npy",
                                                     "unet/output_block/11/output_blocks_11_1_proj_out_bias.npy");
}

void UNetModel::initOut() {
    out_group_norm = new GroupNorm(context, cmdQueue, deviceId, assetManager,
                                   32, 320, 1e-5,
                                   "unet/out/out_group_norm_weight.npy",
                                   "unet/out/out_group_norm_bias.npy");

    out_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                            320, 4, 3, 1, 1,
                            "unet/out/out_conv2d_weight.npy",
                            "unet/out/out_conv2d_bias.npy");
}

UNetModel::~UNetModel() {
    delete time_embed_0;
    delete time_embed_2;
    delete input_block_0_conv2d;
    delete input_block_1_res_block;
    delete input_block_1_spatial;
    delete input_block_2_res_block;
    delete input_block_2_spatial;
    delete input_block_3_conv2d;
    delete input_block_4_res_block;
    delete input_block_4_spatial;
    delete input_block_5_res_block;
    delete input_block_5_spatial;
    delete input_block_6_conv2d;
    delete input_block_7_res_block;
    delete input_block_7_spatial;
    delete input_block_8_res_block;
    delete input_block_8_spatial;
    delete input_block_9_conv2d;
    delete input_block_10_res_block;
    delete input_block_11_res_block;
    delete middle_block_0_res_block;
    delete middle_block_1_spatial;
    delete middle_block_2_res_block;
    delete output_block_0_res_block;
    delete output_block_1_res_block;
    delete output_block_2_res_block;
    delete output_block_2_up_sample;
    delete output_block_3_res_block;
    delete output_block_3_spatial;
    delete output_block_4_res_block;
    delete output_block_4_spatial;
    delete output_block_5_res_block;
    delete output_block_5_spatial;
    delete output_block_5_up_sample;
    delete output_block_6_res_block;
    delete output_block_6_spatial;
    delete output_block_7_res_block;
    delete output_block_7_spatial;
    delete output_block_8_res_block;
    delete output_block_8_spatial;
    delete output_block_8_up_sample;
    delete output_block_9_res_block;
    delete output_block_9_spatial;
    delete output_block_10_res_block;
    delete output_block_10_spatial;
    delete output_block_11_res_block;
    delete output_block_11_spatial;
    delete out_group_norm;
    delete out_conv2d;
    clReleaseKernel(kernel_silu);
}

/*
 * Assume Batch size 'B' is 1.
 * @param x: [B, LATENT_CHANNEL(4), HEIGHT/DOWN_SAMPLING(512/8=64), WIDTH/DOWN_SAMPLING(512/8=64)]
 * @param timestep: long. originally [B]
 * @param condition: [B, CONTEXT_LENGTH(77), EMBEDDING_SIZE(1024)]
 */
std::vector<float> UNetModel::forward(const std::vector<float> &x, long timestep,
                                      const std::vector<float> &condition) {
    cl_int err;
    cl_event event0_0, event0_1, event0_2, event0_3;
    cl_event event1_0, event1_1, event1_2[2], event1_3, event1_4, event1_5, event1_6, event1_7, event1_8, event1_9, event1_10, event1_11;
    cl_event event1_12, event1_13, event1_14, event1_15, event1_16, event1_17, event1_18;
    cl_event event2_0, event2_1, event2_2;
    cl_event event3_0, event3_1, event3_2, event3_3, event3_4, event3_5, event3_6, event3_7, event3_8, event3_9, event3_10, event3_11;
    cl_event event3_12, event3_13, event3_14, event3_15, event3_16, event3_17, event3_18, event3_19, event3_20, event3_21, event3_22;
    cl_event event3_23, event3_24, event3_25, event3_26, event3_27, event3_28, event3_29, event3_30, event3_31, event3_32, event3_33;
    cl_event event3_34, event3_35, event3_36, event3_37, event3_38;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed, bufferCondition;
    cl_mem bufferInput, bufferInput_0, bufferInput_1, bufferInput_2, bufferInput_3, bufferInput_4, bufferInput_5, bufferInput_6, bufferInput_7, bufferInput_8, bufferInput_9, bufferInput_10, bufferInput_11;
    cl_mem buffer_640_32, buffer_1280_16, buffer_1280_8;
    cl_mem buffer_2560_8, buffer_2560_16, buffer_1920_16, buffer_1280_32, buffer_1920_32, buffer_960_32, buffer_640_64, buffer_960_64, buffer_320_64, buffer_4_64;

    /* time_embed layer */
    auto t_emb = timestep_embedding(timestep);

    bufferTimeEmbed = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * t_emb.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferEmbedTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(float) * MODEL_CHANNELS * 4,
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferEmbed = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * MODEL_CHANNELS * 4,
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferTimeEmbed, CL_FALSE, 0,
                               sizeof(float) * t_emb.size(),
                               t_emb.data(), 0, nullptr, &event0_0);
    CHECK_ERROR(err);

    err = time_embed_0->forward(bufferTimeEmbed, bufferEmbedTemp, 1, &event0_0, &event0_1);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferEmbedTemp);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferEmbedTemp);
    CHECK_ERROR(err);

    size_t embedWorkSize[1] = {MODEL_CHANNELS * 4};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 embedWorkSize, nullptr, 1, &event0_1, &event0_2);
    CHECK_ERROR(err);

    err = time_embed_2->forward(bufferEmbedTemp, bufferEmbed, 1, &event0_2, &event0_3);
    CHECK_ERROR(err);

    // timestep=981. max diff: 0.00000476837158203125
    // util::testBuffer(cmdQueue, bufferEmbed, "unet/time_embed/test/test_time_embed.npy");
    /* time_embed layer */

    /* input_block layer */
    /* input_block layer[0] */
    initInputBlock0();
    bufferInput_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    bufferInput = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * x.size(),
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferInput, CL_FALSE, 0,
                               sizeof(float) * x.size(),
                               x.data(), 0, nullptr, &event1_0);
    CHECK_ERROR(err);


    input_block_0_conv2d->init();
    err = input_block_0_conv2d->forward(bufferInput, bufferInput_0, 1, &event1_0, &event1_1);
    CHECK_ERROR(err);
    delete input_block_0_conv2d;

    // x=seed45.npy. timestep=981. max diff: 0.00000059604644775391
    // util::testBuffer(cmdQueue, bufferInput_0, "unet/input_block/test/test_input_block_0_conv2d.npy");
    /* input_block layer[0] */

    /* input_block layer[1] */
    initInputBlock1();
    bufferInput_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    bufferCondition = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * condition.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferCondition, CL_FALSE, 0,
                               sizeof(float) * condition.size(),
                               condition.data(), 0, nullptr, &event1_2[0]);
    CHECK_ERROR(err);

    input_block_1_res_block->init();
    err = input_block_1_res_block->forward(bufferInput_0, bufferEmbed, bufferInput_1,
                                           1, &event0_3,
                                           1, &event1_1, &event1_2[1]);
    CHECK_ERROR(err);
    delete input_block_1_res_block;

    input_block_1_spatial->init();
    err = input_block_1_spatial->forward(bufferInput_1, bufferCondition, bufferInput_1,
                                         2, event1_2, &event1_3);
    CHECK_ERROR(err);
    delete input_block_1_spatial;
    /* input_block layer[1] */

    /* input_block layer[2] */
    initInputBlock2();
    bufferInput_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_2_res_block->init();
    err = input_block_2_res_block->forward(bufferInput_1, bufferEmbed, bufferInput_2,
                                           1, &event0_3,
                                           1, &event1_3, &event1_4);
    CHECK_ERROR(err);
    delete input_block_2_res_block;

    // test_input_block_2_res.npy max diff: 0.00001192092895507812
    // util::testBuffer(cmdQueue, bufferInput_2, "unet/input_block/test/test_input_block_2_res.npy");

    input_block_2_spatial->init();
    err = input_block_2_spatial->forward(bufferInput_2, bufferCondition, bufferInput_2,
                                         1, &event1_4, &event1_5);
    CHECK_ERROR(err);
    delete input_block_2_spatial;

    // max diff: 0.00001168251037597656
    // util::testBuffer(cmdQueue, bufferInput_2, "unet/input_block/test/test_input_block_2.npy");
    /* input_block layer[2] */

    /* input_block layer[3] */
    initInputBlock3();
    bufferInput_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_3_conv2d->init();
    err = input_block_3_conv2d->forward(bufferInput_2, bufferInput_3,
                                        1, &event1_5, &event1_6);
    CHECK_ERROR(err);
    delete input_block_3_conv2d;

    // max diff: 0.00001525878906250000
    // util::testBuffer(cmdQueue, bufferInput_3, "unet/input_block/test/test_input_block_3.npy");
    /* input_block layer[3] */

    /* input_block layer[4] */
    initInputBlock4();
    bufferInput_4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_4_res_block->init();
    err = input_block_4_res_block->forward(bufferInput_3, bufferEmbed, bufferInput_4,
                                           1, &event0_3,
                                           1, &event1_6, &event1_7);
    CHECK_ERROR(err);
    delete input_block_4_res_block;

    // max diff: 0.00001382827758789062
    // util::testBuffer(cmdQueue, bufferInput_4, "unet/input_block/test/test_input_block_4_res.npy");

    input_block_4_spatial->init();
    err = input_block_4_spatial->forward(bufferInput_4, bufferCondition, bufferInput_4,
                                         1, &event1_7, &event1_8);
    CHECK_ERROR(err);
    delete input_block_4_spatial;

    // max diff: 0.00002956390380859375
    // util::testBuffer(cmdQueue, bufferInput_4, "unet/input_block/test/test_input_block_4.npy");
    /* input_block layer[4] */

    /* input_block layer[5] */
    initInputBlock5();
    bufferInput_5 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_5_res_block->init();
    err = input_block_5_res_block->forward(bufferInput_4, bufferEmbed, bufferInput_5,
                                           1, &event0_3,
                                           1, &event1_8, &event1_9);
    CHECK_ERROR(err);
    delete input_block_5_res_block;

    input_block_5_spatial->init();
    err = input_block_5_spatial->forward(bufferInput_5, bufferCondition, bufferInput_5,
                                         1, &event1_9, &event1_10);
    CHECK_ERROR(err);
    delete input_block_5_spatial;

    // max diff: 0.00013732910156250000
    // util::testBuffer(cmdQueue, bufferInput_5, "unet/input_block/test/test_input_block_5.npy");

    /* input_block layer[5] */

    /* input_block layer[6] */
    initInputBlock6();
    bufferInput_6 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 16 * 16,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_6_conv2d->init();
    err = input_block_6_conv2d->forward(bufferInput_5, bufferInput_6,
                                        1, &event1_10, &event1_11);
    CHECK_ERROR(err);
    delete input_block_6_conv2d;

    // max diff: 0.00004652142524719238
    // util::testBuffer(cmdQueue, bufferInput_6, "unet/input_block/test/test_input_block_6.npy");
    /* input_block layer[6] */

    /* input_block layer[7] */
    initInputBlock7();
    bufferInput_7 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 16 * 16,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_7_res_block->init();
    err = input_block_7_res_block->forward(bufferInput_6, bufferEmbed, bufferInput_7,
                                           1, &event0_3,
                                           1, &event1_11, &event1_12);
    CHECK_ERROR(err);
    delete input_block_7_res_block;

    input_block_7_spatial->init();
    err = input_block_7_spatial->forward(bufferInput_7, bufferCondition, bufferInput_7,
                                         1, &event1_12, &event1_13);
    CHECK_ERROR(err);
    delete input_block_7_spatial;

    // max diff: 0.00004172325134277344
    // util::testBuffer(cmdQueue, bufferInput_7, "unet/input_block/test/test_input_block_7.npy");
    /* input_block layer[7] */

    /* input_block layer[8] */
    initInputBlock8();
    bufferInput_8 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 16 * 16,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_8_res_block->init();
    err = input_block_8_res_block->forward(bufferInput_7, bufferEmbed, bufferInput_8,
                                           1, &event0_3,
                                           1, &event1_13, &event1_14);
    CHECK_ERROR(err);
    delete input_block_8_res_block;

    input_block_8_spatial->init();
    err = input_block_8_spatial->forward(bufferInput_8, bufferCondition, bufferInput_8,
                                         1, &event1_14, &event1_15);
    CHECK_ERROR(err);
    delete input_block_8_spatial;

    // max diff: 0.00004684925079345703
    // util::testBuffer(cmdQueue, bufferInput_8, "unet/input_block/test/test_input_block_8.npy");
    /* input_block layer[8] */

    /* input_block layer[9] */
    initInputBlock9();
    bufferInput_9 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                   nullptr, &err);
    CHECK_ERROR(err);

    input_block_9_conv2d->init();
    err = input_block_9_conv2d->forward(bufferInput_8, bufferInput_9,
                                        1, &event1_15, &event1_16);
    CHECK_ERROR(err);
    delete input_block_9_conv2d;
    /* input_block layer[9] */

    /* input_block layer[10] */
    initInputBlock10();
    bufferInput_10 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                    nullptr, &err);
    CHECK_ERROR(err);

    input_block_10_res_block->init();
    err = input_block_10_res_block->forward(bufferInput_9, bufferEmbed, bufferInput_10,
                                            1, &event0_3,
                                            1, &event1_16, &event1_17);
    CHECK_ERROR(err);
    delete input_block_10_res_block;
    /* input_block layer[10] */

    /* input_block layer[11] */
    initInputBlock11();
    bufferInput_11 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                    nullptr, &err);
    CHECK_ERROR(err);

    input_block_11_res_block->init();
    err = input_block_11_res_block->forward(bufferInput_10, bufferEmbed, bufferInput_11,
                                            1, &event0_3,
                                            1, &event1_17, &event1_18);
    CHECK_ERROR(err);
    delete input_block_11_res_block;

    // max diff: 0.00013542175292968750
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/input_block/test/test_input_block_11.npy");
    /* input_block layer[11] */
    /* input_block layer */

    /* middle_block layer */
    initMiddleBlock();
    buffer_1280_8 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                   nullptr, &err);
    CHECK_ERROR(err);

    middle_block_0_res_block->init();
    err = middle_block_0_res_block->forward(bufferInput_11, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event1_18, &event2_0);
    CHECK_ERROR(err);
    delete middle_block_0_res_block;

    middle_block_1_spatial->init();
    err = middle_block_1_spatial->forward(buffer_1280_8, bufferCondition, buffer_1280_8,
                                          1, &event2_0, &event2_1);
    CHECK_ERROR(err);
    delete middle_block_1_spatial;

    middle_block_2_res_block->init();
    err = middle_block_2_res_block->forward(buffer_1280_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event2_1, &event2_2);
    CHECK_ERROR(err);
    delete middle_block_2_res_block;

    // max diff: 0.00013828277587890625
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/middle_block/test/test_middle_block.npy");
    /* middle_block layer */

    /* output_block layer */
    /* output_block layer[0] */
    initOutputBlock0();
    buffer_2560_8 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 8 * MODEL_CHANNELS * 8 * 8,
                                   nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_1280_8, bufferInput_11, buffer_2560_8,
                  1, &event2_2, &event3_0);

    output_block_0_res_block->init();
    err = output_block_0_res_block->forward(buffer_2560_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event3_0, &event3_1);
    CHECK_ERROR(err);
    delete output_block_0_res_block;
    // test_output_block_0.npy max diff: 0.00009822845458984375
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/output_block/test/test_output_block_0.npy");
    /* output_block layer[0] */

    /* output_block layer[1] */
    initOutputBlock1();
    concat_buffer(buffer_1280_8, bufferInput_10, buffer_2560_8,
                  1, &event3_1, &event3_2);

    output_block_1_res_block->init();
    err = output_block_1_res_block->forward(buffer_2560_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event3_2, &event3_3);
    CHECK_ERROR(err);
    delete output_block_1_res_block;

    // test_output_block_1.npy max diff: 0.00010108947753906250
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/output_block/test/test_output_block_1.npy");
    /* output_block layer[1] */

    /* output_block layer[2] */
    initOutputBlock2();
    buffer_1280_16 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 4 * MODEL_CHANNELS * 16 * 16,
                                    nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_1280_8, bufferInput_9, buffer_2560_8,
                  1, &event3_3, &event3_4);

    output_block_2_res_block->init();
    err = output_block_2_res_block->forward(buffer_2560_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event3_4, &event3_5);
    CHECK_ERROR(err);
    delete output_block_2_res_block;

    output_block_2_up_sample->init();
    err = output_block_2_up_sample->forward(buffer_1280_8, buffer_1280_16,
                                            1, &event3_5, &event3_6);
    CHECK_ERROR(err);
    delete output_block_2_up_sample;

    // test_output_block_2.npy max diff: 0.00007247924804687500
    // util::testBuffer(cmdQueue, buffer_1280_16, "unet/output_block/test/test_output_block_2.npy");
    /* output_block layer[2] */

    /* output_block layer[3] */
    initOutputBlock3();
    buffer_2560_16 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 8 * MODEL_CHANNELS * 16 * 16,
                                    nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_1280_16, bufferInput_8, buffer_2560_16,
                  1, &event3_6, &event3_7);

    output_block_3_res_block->init();
    err = output_block_3_res_block->forward(buffer_2560_16, bufferEmbed, buffer_1280_16,
                                            1, &event0_3,
                                            1, &event3_7, &event3_8);
    CHECK_ERROR(err);
    delete output_block_3_res_block;

    output_block_3_spatial->init();
    err = output_block_3_spatial->forward(buffer_1280_16, bufferCondition, buffer_1280_16,
                                          1, &event3_8, &event3_9);
    CHECK_ERROR(err);
    delete output_block_3_spatial;

    // test_output_block_3.npy max diff: 0.00009536743164062500
    // util::testBuffer(cmdQueue, buffer_1280_16, "unet/output_block/test/test_output_block_3.npy");
    /* output_block layer[3] */

    /* output_block layer[4] */
    initOutputBlock4();

    concat_buffer(buffer_1280_16, bufferInput_7, buffer_2560_16,
                  1, &event3_9, &event3_10);

    output_block_4_res_block->init();
    err = output_block_4_res_block->forward(buffer_2560_16, bufferEmbed, buffer_1280_16,
                                            1, &event0_3,
                                            1, &event3_10, &event3_11);
    CHECK_ERROR(err);
    delete output_block_4_res_block;

    output_block_4_spatial->init();
    err = output_block_4_spatial->forward(buffer_1280_16, bufferCondition, buffer_1280_16,
                                          1, &event3_11, &event3_12);
    CHECK_ERROR(err);
    delete output_block_4_spatial;
    /* output_block layer[4] */

    /* output_block layer[5] */
    initOutputBlock5();

    buffer_1920_16 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 6 * MODEL_CHANNELS * 16 * 16,
                                    nullptr, &err);
    CHECK_ERROR(err);

    buffer_1280_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 4 * MODEL_CHANNELS * 32 * 32,
                                    nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_1280_16, bufferInput_6, buffer_1920_16,
                  1, &event3_12, &event3_13);

    output_block_5_res_block->init();
    err = output_block_5_res_block->forward(buffer_1920_16, bufferEmbed, buffer_1280_16,
                                            1, &event0_3,
                                            1, &event3_13, &event3_14);
    CHECK_ERROR(err);
    delete output_block_5_res_block;

    output_block_5_spatial->init();
    err = output_block_5_spatial->forward(buffer_1280_16, bufferCondition, buffer_1280_16,
                                          1, &event3_14, &event3_15);
    CHECK_ERROR(err);
    delete output_block_5_spatial;

    output_block_5_up_sample->init();
    err = output_block_5_up_sample->forward(buffer_1280_16, buffer_1280_32,
                                            1, &event3_15, &event3_16);
    CHECK_ERROR(err);
    delete output_block_5_up_sample;
    /* output_block layer[5] */

    /* output_block layer[6] */
    initOutputBlock6();
    buffer_1920_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 6 * MODEL_CHANNELS * 32 * 32,
                                    nullptr, &err);
    CHECK_ERROR(err);

    buffer_640_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_1280_32, bufferInput_5, buffer_1920_32,
                  1, &event3_16, &event3_17);

    output_block_6_res_block->init();
    err = output_block_6_res_block->forward(buffer_1920_32, bufferEmbed, buffer_640_32,
                                            1, &event0_3,
                                            1, &event3_17, &event3_18);
    CHECK_ERROR(err);
    delete output_block_6_res_block;

    output_block_6_spatial->init();
    err = output_block_6_spatial->forward(buffer_640_32, bufferCondition, buffer_640_32,
                                          1, &event3_18, &event3_19);
    CHECK_ERROR(err);
    delete output_block_6_spatial;
    /* output_block layer[6] */

    /* output_block layer[7] */
    initOutputBlock7();

    concat_buffer(buffer_640_32, bufferInput_4, buffer_1280_32,
                  1, &event3_19, &event3_20);

    output_block_7_res_block->init();
    err = output_block_7_res_block->forward(buffer_1280_32, bufferEmbed, buffer_640_32,
                                            1, &event0_3,
                                            1, &event3_20, &event3_21);
    CHECK_ERROR(err);
    delete output_block_7_res_block;

    output_block_7_spatial->init();
    err = output_block_7_spatial->forward(buffer_640_32, bufferCondition, buffer_640_32,
                                          1, &event3_21, &event3_22);
    CHECK_ERROR(err);
    delete output_block_7_spatial;
    /* output_block layer[7] */

    /* output_block layer[8] */
    initOutputBlock8();

    buffer_960_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 3 * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    buffer_640_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_640_32, bufferInput_3, buffer_960_32,
                  1, &event3_22, &event3_23);

    output_block_8_res_block->init();
    err = output_block_8_res_block->forward(buffer_960_32, bufferEmbed, buffer_640_32,
                                            1, &event0_3,
                                            1, &event3_23, &event3_24);
    CHECK_ERROR(err);
    delete output_block_8_res_block;

    output_block_8_spatial->init();
    err = output_block_8_spatial->forward(buffer_640_32, bufferCondition, buffer_640_32,
                                          1, &event3_24, &event3_25);
    CHECK_ERROR(err);
    delete output_block_8_spatial;

    output_block_8_up_sample->init();
    err = output_block_8_up_sample->forward(buffer_640_32, buffer_640_64,
                                            1, &event3_25, &event3_26);
    CHECK_ERROR(err);
    delete output_block_8_up_sample;
    /* output_block layer[8] */

    /* output_block layer[9] */
    initOutputBlock9();

    buffer_960_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 3 * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    buffer_320_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    concat_buffer(buffer_640_64, bufferInput_2, buffer_960_64,
                  1, &event3_26, &event3_27);

    output_block_9_res_block->init();
    err = output_block_9_res_block->forward(buffer_960_64, bufferEmbed, buffer_320_64,
                                            1, &event0_3,
                                            1, &event3_27, &event3_28);
    CHECK_ERROR(err);
    delete output_block_9_res_block;

    output_block_9_spatial->init();
    err = output_block_9_spatial->forward(buffer_320_64, bufferCondition, buffer_320_64,
                                          1, &event3_28, &event3_29);
    CHECK_ERROR(err);
    delete output_block_9_spatial;
    /* output_block layer[9] */

    /* output_block layer[10] */
    initOutputBlock10();

    concat_buffer(buffer_320_64, bufferInput_1, buffer_640_64,
                  1, &event3_29, &event3_30);

    output_block_10_res_block->init();
    err = output_block_10_res_block->forward(buffer_640_64, bufferEmbed, buffer_320_64,
                                             1, &event0_3,
                                             1, &event3_30, &event3_31);
    CHECK_ERROR(err);
    delete output_block_10_res_block;

    output_block_10_spatial->init();
    err = output_block_10_spatial->forward(buffer_320_64, bufferCondition, buffer_320_64,
                                           1, &event3_31, &event3_32);
    CHECK_ERROR(err);
    delete output_block_10_spatial;
    /* output_block layer[10] */

    /* output_block layer[11] */
    initOutputBlock11();

    concat_buffer(buffer_320_64, bufferInput_0, buffer_640_64,
                  1, &event3_32, &event3_33);

    output_block_11_res_block->init();
    err = output_block_11_res_block->forward(buffer_640_64, bufferEmbed, buffer_320_64,
                                             1, &event0_3,
                                             1, &event3_33, &event3_34);
    CHECK_ERROR(err);
    delete output_block_11_res_block;

    output_block_11_spatial->init();
    err = output_block_11_spatial->forward(buffer_320_64, bufferCondition, buffer_320_64,
                                           1, &event3_34, &event3_35);
    CHECK_ERROR(err);
    delete output_block_11_spatial;

    // test_output_block_11.npy max diff: 0.00001716613769531250
    // util::testBuffer(cmdQueue, buffer_320_64, "unet/output_block/test/test_output_block_11.npy");
    /* output_block layer[11] */
    /* output_block layer */

    /* out */
    initOut();

    buffer_4_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * 4 * 64 * 64,
                                 nullptr, &err);
    CHECK_ERROR(err);

    out_group_norm->init();
    err = out_group_norm->forward(buffer_320_64, buffer_320_64,
                                  1, &event3_35, &event3_36);
    CHECK_ERROR(err);
    delete out_group_norm;

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &buffer_320_64);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &buffer_320_64);
    CHECK_ERROR(err);

    size_t outSiluSize[1] = {MODEL_CHANNELS * 64 * 64};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 outSiluSize, nullptr, 1, &event3_36, &event3_37);
    CHECK_ERROR(err);

    out_conv2d->init();
    err = out_conv2d->forward(buffer_320_64, buffer_4_64,
                              1, &event3_37, &event3_38);
    CHECK_ERROR(err);
    delete out_conv2d;

    // util::testBuffer(cmdQueue, buffer_4_64, "unet/out/test/test_out.npy");
    /* out */

    /* result */
    std::vector<float> result(4 * 64 * 64);
    err = clEnqueueReadBuffer(cmdQueue, buffer_4_64, CL_TRUE, 0,
                              sizeof(float) * result.size(),
                              result.data(), 1, &event3_38, nullptr);
    CHECK_ERROR(err);
    /* result */

    clFinish(cmdQueue);

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Finished");

    clReleaseEvent(event0_0);
    clReleaseEvent(event0_1);
    clReleaseEvent(event0_2);
    clReleaseEvent(event0_3);
    clReleaseEvent(event1_0);
    clReleaseEvent(event1_1);
    clReleaseEvent(event1_2[0]);
    clReleaseEvent(event1_2[1]);
    clReleaseEvent(event1_3);
    clReleaseEvent(event1_4);
    clReleaseEvent(event1_5);
    clReleaseEvent(event1_6);
    clReleaseEvent(event1_7);
    clReleaseEvent(event1_8);
    clReleaseEvent(event1_9);
    clReleaseEvent(event1_10);
    clReleaseEvent(event1_11);
    clReleaseEvent(event1_12);
    clReleaseEvent(event1_13);
    clReleaseEvent(event1_14);
    clReleaseEvent(event1_15);
    clReleaseEvent(event1_16);
    clReleaseEvent(event1_17);
    clReleaseEvent(event1_18);
    clReleaseEvent(event2_0);
    clReleaseEvent(event2_1);
    clReleaseEvent(event2_2);
    clReleaseEvent(event3_0);
    clReleaseEvent(event3_1);
    clReleaseEvent(event3_2);
    clReleaseEvent(event3_3);
    clReleaseEvent(event3_4);
    clReleaseEvent(event3_5);
    clReleaseEvent(event3_6);
    clReleaseEvent(event3_7);
    clReleaseEvent(event3_8);
    clReleaseEvent(event3_9);
    clReleaseEvent(event3_10);
    clReleaseEvent(event3_11);
    clReleaseEvent(event3_12);
    clReleaseEvent(event3_13);
    clReleaseEvent(event3_14);
    clReleaseEvent(event3_15);
    clReleaseEvent(event3_16);
    clReleaseEvent(event3_17);
    clReleaseEvent(event3_18);
    clReleaseEvent(event3_19);
    clReleaseEvent(event3_20);
    clReleaseEvent(event3_21);
    clReleaseEvent(event3_22);
    clReleaseEvent(event3_23);
    clReleaseEvent(event3_24);
    clReleaseEvent(event3_25);
    clReleaseEvent(event3_26);
    clReleaseEvent(event3_27);
    clReleaseEvent(event3_28);
    clReleaseEvent(event3_29);
    clReleaseEvent(event3_30);
    clReleaseEvent(event3_31);
    clReleaseEvent(event3_32);
    clReleaseEvent(event3_33);
    clReleaseEvent(event3_34);
    clReleaseEvent(event3_35);
    clReleaseEvent(event3_36);
    clReleaseEvent(event3_37);
    clReleaseEvent(event3_38);
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);
    clReleaseMemObject(bufferInput);
    clReleaseMemObject(buffer_320_64);
    clReleaseMemObject(bufferCondition);
    clReleaseMemObject(buffer_640_32);
    clReleaseMemObject(buffer_1280_16);
    clReleaseMemObject(buffer_1280_8);
    clReleaseMemObject(bufferInput_0);
    clReleaseMemObject(bufferInput_1);
    clReleaseMemObject(bufferInput_2);
    clReleaseMemObject(bufferInput_3);
    clReleaseMemObject(bufferInput_4);
    clReleaseMemObject(bufferInput_5);
    clReleaseMemObject(bufferInput_6);
    clReleaseMemObject(bufferInput_7);
    clReleaseMemObject(bufferInput_8);
    clReleaseMemObject(bufferInput_9);
    clReleaseMemObject(bufferInput_10);
    clReleaseMemObject(bufferInput_11);
    clReleaseMemObject(buffer_2560_8);
    clReleaseMemObject(buffer_2560_16);
    clReleaseMemObject(buffer_1920_16);
    clReleaseMemObject(buffer_1280_32);
    clReleaseMemObject(buffer_1920_32);
    clReleaseMemObject(buffer_960_32);
    clReleaseMemObject(buffer_640_64);
    clReleaseMemObject(buffer_960_64);
    clReleaseMemObject(buffer_4_64);

    return result;
}

/*
 * @param timestep: long. originally [B]
 * @return [B, MODEL_CHANNELS(320)]
 */
std::vector<float> UNetModel::timestep_embedding(long timestep) {
    float max_period = 10000.0f;
    int half = MODEL_CHANNELS / 2;

    std::vector<float> embedding(MODEL_CHANNELS);
    for (int i = 0; i < half; i++) {
        auto freq = exp((-log(max_period)) * static_cast<float>(i) / static_cast<float>(half));
        auto arg = static_cast<float>(timestep) * freq;
        embedding[i] = cos(arg);
        embedding[i + half] = sin(arg);
    }

    // timestep=981. max diff: 0.00000005960464477539
    // util::testBuffer(embedding, "unet/time_embed/test/test_timestep_embedding.npy");
    return embedding;
}

void
UNetModel::concat_buffer(
        cl_mem input1, cl_mem input2, cl_mem output, cl_uint num_events_in_list,
        const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0;

    size_t input1_bytes, input2_bytes, output_bytes;
    err = clGetMemObjectInfo(input1, CL_MEM_SIZE, sizeof(size_t), &input1_bytes, nullptr);
    CHECK_ERROR(err);
    err = clGetMemObjectInfo(input2, CL_MEM_SIZE, sizeof(size_t), &input2_bytes, nullptr);
    CHECK_ERROR(err);
    err = clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(size_t), &output_bytes, nullptr);
    CHECK_ERROR(err);

    if (input1_bytes + input2_bytes != output_bytes) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "concat_buffer: input1_size(%ld) + input2_size(%ld) != output_size(%ld)",
                            input1_bytes, input2_bytes, output_bytes);
        throw std::runtime_error("concat_buffer: input1_size + input2_size != output_size");
    }

    err = clEnqueueCopyBuffer(cmdQueue, input1, output, 0, 0, input1_bytes,
                              num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR(err);

    err = clEnqueueCopyBuffer(cmdQueue, input2, output, 0, input1_bytes, input2_bytes,
                              1, &event0, event);
    CHECK_ERROR(err);
}

void
UNetModel::test(const std::vector<float> &x, long timestep, const std::vector<float> &condition) {
    cl_int err;
    cl_event event0, event1, event2, event3, event4, event5;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed, bufferCondition, bufferInput, buffer_320_64, buffer_4_64;

    auto t_emb = timestep_embedding(timestep);

    bufferTimeEmbed = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * t_emb.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferEmbed = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * MODEL_CHANNELS * 4,
                                 nullptr, &err);
    CHECK_ERROR(err);

    bufferCondition = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * condition.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferInput = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * x.size(),
                                 nullptr, &err);
    CHECK_ERROR(err);

    buffer_320_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    buffer_4_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * 4 * 64 * 64,
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferTimeEmbed, CL_FALSE, 0,
                               sizeof(float) * t_emb.size(),
                               t_emb.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferCondition, CL_FALSE, 0,
                               sizeof(float) * condition.size(),
                               condition.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferInput, CL_FALSE, 0,
                               sizeof(float) * x.size(),
                               x.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    bufferEmbedTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(float) * MODEL_CHANNELS * 4,
                                     nullptr, &err);
    CHECK_ERROR(err);

    err = time_embed_0->forward(bufferTimeEmbed, bufferEmbedTemp, 0, nullptr, &event0);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferEmbedTemp);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferEmbedTemp);
    CHECK_ERROR(err);

    size_t embedWorkSize[1] = {MODEL_CHANNELS * 4};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 embedWorkSize, nullptr, 1, &event0, &event1);
    CHECK_ERROR(err);

    err = time_embed_2->forward(bufferEmbedTemp, bufferEmbed, 1, &event1, &event2);
    CHECK_ERROR(err);

    initOut();
    out_group_norm->init();
    err = out_group_norm->forward(bufferInput, buffer_320_64, 0, nullptr, &event3);
    CHECK_ERROR(err);

    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &buffer_320_64);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &buffer_320_64);
    CHECK_ERROR(err);

    size_t outSiluSize[1] = {MODEL_CHANNELS * 64 * 64};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 outSiluSize, nullptr, 1, &event3, &event4);
    CHECK_ERROR(err);

    out_conv2d->init();
    err = out_conv2d->forward(buffer_320_64, buffer_4_64,
                              1, &event4, nullptr);
    CHECK_ERROR(err);
    util::testBuffer(cmdQueue, buffer_4_64, "unet/out/test/test_out.npy");

    // test_output_block_0 max diff: 0.00002098083496093750
    // test_output_block_1.npy max diff: 0.00002479553222656250
    // test_output_block_2.npy max diff: 0.00007152557373046875
    // test_output_block_3.npy max diff: 0.00005149841308593750
    // test_output_block_4.npy max diff: 0.00006866455078125000
    // test_output_block_5.npy max diff: 0.00042724609375000000
    // test_output_block_6.npy max diff: 0.00018310546875000000
    // test_output_block_7.npy max diff: 0.00008010864257812500
    // test_output_block_8.npy max diff: 0.00036621093750000000
    // test_output_block_9.npy max diff: 0.00001406669616699219
    // test_output_block_10.npy max diff: 0.00000619888305664062
    // test_output_block_11.npy max diff: 0.00001430511474609375
    // test_out.npy max diff: 0.00000190734863281250
    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);
    clReleaseMemObject(bufferInput);
    clReleaseMemObject(bufferCondition);
    clReleaseMemObject(buffer_320_64);
    clReleaseMemObject(buffer_4_64);
}
