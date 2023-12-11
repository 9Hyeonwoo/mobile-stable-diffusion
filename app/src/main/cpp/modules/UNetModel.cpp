//
// Created by 구현우 on 2023/12/07.
//

#include "UNetModel.h"

#include "util.h"
#include <android/log.h>

#define LOG_TAG "UNET_MODEL"
#define MODEL_CHANNELS 320

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
) : context(context), cmdQueue(cmdQueue) {
    cl_int err;
    time_embed_0 = new Linear(context, cmdQueue, deviceId, assetManager,
                              "unet/time_embed/time_embed_0_weight.npy",
                              "unet/time_embed/time_embed_0_bias.npy");

    time_embed_2 = new Linear(context, cmdQueue, deviceId, assetManager,
                              "unet/time_embed/time_embed_2_weight.npy",
                              "unet/time_embed/time_embed_2_bias.npy");

    input_block_0_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      "unet/input_block/0/input_block_0_conv2d_weight.npy",
                                      "unet/input_block/0/input_block_0_conv2d_bias.npy",
                                      1, 1);

    input_block_1_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, 320,
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
                                                   320, 5, 64,
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

    input_block_2_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, 320,
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
                                                   320, 5, 64,
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

    input_block_3_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      "unet/input_block/3/input_blocks_3_0_op_weight.npy",
                                      "unet/input_block/3/input_blocks_3_0_op_bias.npy",
                                      2, 1);

    input_block_4_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           320, 640,
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
                                                   640, 10, 64,
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

    input_block_5_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           640, 640,
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
                                                   640, 10, 64,
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

    input_block_6_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      "unet/input_block/6/input_blocks_6_0_op_weight.npy",
                                      "unet/input_block/6/input_blocks_6_0_op_bias.npy",
                                      2, 1);

    input_block_7_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           640, 1280,
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
                                                   1280, 20, 64,
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

    input_block_8_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                           1280, 1280,
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
                                                   1280, 20, 64,
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

    input_block_9_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                      "unet/input_block/9/input_blocks_9_0_op_weight.npy",
                                      "unet/input_block/9/input_blocks_9_0_op_bias.npy",
                                      2, 1);

    input_block_10_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, 1280,
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

    input_block_11_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, 1280,
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

    middle_block_0_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            1280, 1280,
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
                                                    1280, 20, 64,
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
                                            1280, 1280,
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
    output_block_0_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, 1280,
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

    output_block_1_res_block = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                            2560, 1280,
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

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR(err);

    clReleaseProgram(program);
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
    cl_event event1_0, event1_1, event1_2, event1_3, event1_4, event1_5, event1_6, event1_7, event1_8, event1_9, event1_10, event1_11;
    cl_event event1_12, event1_13, event1_14, event1_15, event1_16, event1_17, event1_18;
    cl_event event2_0, event2_1, event2_2;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed;
    cl_mem bufferInput, buffer_320_64, bufferCondition, buffer_320_32;
    cl_mem buffer_640_32, buffer_640_16, buffer_1280_16, buffer_1280_8;

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

    bufferInput = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * x.size(),
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferInput, CL_FALSE, 0,
                               sizeof(float) * x.size(),
                               x.data(), 0, nullptr, &event1_0);
    CHECK_ERROR(err);

    buffer_320_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_0_conv2d->forward(bufferInput, buffer_320_64, 1, &event1_0, &event1_1);
    CHECK_ERROR(err);

    // x=seed45.npy. timestep=981. max diff: 0.00000059604644775391
    // util::testBuffer(cmdQueue, buffer_320_64, "unet/input_block/test/test_input_block_0_conv2d.npy");
    /* input_block layer[0] */

    /* input_block layer[1] */

    bufferCondition = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * condition.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferCondition, CL_FALSE, 0,
                               sizeof(float) * condition.size(),
                               condition.data(), 0, nullptr, &event1_2);

    err = input_block_1_res_block->forward(buffer_320_64, bufferEmbed, buffer_320_64,
                                           1, &event0_3,
                                           1, &event1_1, &event1_2);
    CHECK_ERROR(err);

    err = input_block_1_spatial->forward(buffer_320_64, bufferCondition, buffer_320_64,
                                         2, &event1_2, &event1_3);
    CHECK_ERROR(err);
    /* input_block layer[1] */

    /* input_block layer[2] */
    err = input_block_2_res_block->forward(buffer_320_64, bufferEmbed, buffer_320_64,
                                           1, &event0_3,
                                           1, &event1_3, &event1_4);
    CHECK_ERROR(err);

    err = input_block_2_spatial->forward(buffer_320_64, bufferCondition, buffer_320_64,
                                         1, &event1_4, &event1_5);
    CHECK_ERROR(err);

    // max diff: 0.00001168251037597656
    // util::testBuffer(cmdQueue, buffer_320_64, "unet/input_block/test/test_input_block_2.npy");
    /* input_block layer[2] */

    /* input_block layer[3] */
    buffer_320_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_3_conv2d->forward(buffer_320_64, buffer_320_32,
                                        1, &event1_5, &event1_6);
    CHECK_ERROR(err);

    // max diff: 0.00001525878906250000
    // util::testBuffer(cmdQueue, buffer_320_32, "unet/input_block/test/test_input_block_3.npy");
    /* input_block layer[3] */

    /* input_block layer[4] */
    buffer_640_32 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 32 * 32,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_4_res_block->forward(buffer_320_32, bufferEmbed, buffer_640_32,
                                           1, &event0_3,
                                           1, &event1_6, &event1_7);
    CHECK_ERROR(err);

    // max diff: 0.00001382827758789062
    // util::testBuffer(cmdQueue, buffer_640_32, "unet/input_block/test/test_input_block_4_res.npy");

    err = input_block_4_spatial->forward(buffer_640_32, bufferCondition, buffer_640_32,
                                         1, &event1_7, &event1_8);
    CHECK_ERROR(err);

    // max diff: 0.00002956390380859375
    // util::testBuffer(cmdQueue, buffer_640_32, "unet/input_block/test/test_input_block_4.npy");
    /* input_block layer[4] */

    /* input_block layer[5] */
    err = input_block_5_res_block->forward(buffer_640_32, bufferEmbed, buffer_640_32,
                                           1, &event0_3,
                                           1, &event1_8, &event1_9);
    CHECK_ERROR(err);

    err = input_block_5_spatial->forward(buffer_640_32, bufferCondition, buffer_640_32,
                                         1, &event1_9, &event1_10);
    CHECK_ERROR(err);

    // max diff: 0.00013732910156250000
    // util::testBuffer(cmdQueue, buffer_640_32, "unet/input_block/test/test_input_block_5.npy");

    /* input_block layer[5] */

    /* input_block layer[6] */
    buffer_640_16 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 2 * MODEL_CHANNELS * 16 * 16,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_6_conv2d->forward(buffer_640_32, buffer_640_16,
                                        1, &event1_10, &event1_11);
    CHECK_ERROR(err);

    // max diff: 0.00004652142524719238
    // util::testBuffer(cmdQueue, buffer_640_16, "unet/input_block/test/test_input_block_6.npy");
    /* input_block layer[6] */

    /* input_block layer[7] */
    buffer_1280_16 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 4 * MODEL_CHANNELS * 16 * 16,
                                    nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_7_res_block->forward(buffer_640_16, bufferEmbed, buffer_1280_16,
                                           1, &event0_3,
                                           1, &event1_11, &event1_12);
    CHECK_ERROR(err);

    err = input_block_7_spatial->forward(buffer_1280_16, bufferCondition, buffer_1280_16,
                                         1, &event1_12, &event1_13);
    CHECK_ERROR(err);

    // max diff: 0.00004172325134277344
    // util::testBuffer(cmdQueue, buffer_1280_16, "unet/input_block/test/test_input_block_7.npy");
    /* input_block layer[7] */

    /* input_block layer[8] */
    err = input_block_8_res_block->forward(buffer_1280_16, bufferEmbed, buffer_1280_16,
                                           1, &event0_3,
                                           1, &event1_13, &event1_14);
    CHECK_ERROR(err);

    err = input_block_8_spatial->forward(buffer_1280_16, bufferCondition, buffer_1280_16,
                                         1, &event1_14, &event1_15);
    CHECK_ERROR(err);

    // max diff: 0.00004684925079345703
    // util::testBuffer(cmdQueue, buffer_1280_16, "unet/input_block/test/test_input_block_8.npy");
    /* input_block layer[8] */

    /* input_block layer[9] */
    buffer_1280_8 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_9_conv2d->forward(buffer_1280_16, buffer_1280_8,
                                        1, &event1_15, &event1_16);
    CHECK_ERROR(err);
    /* input_block layer[9] */

    /* input_block layer[10] */
    err = input_block_10_res_block->forward(buffer_1280_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event1_16, &event1_17);
    CHECK_ERROR(err);
    /* input_block layer[10] */

    /* input_block layer[11] */
    err = input_block_11_res_block->forward(buffer_1280_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event1_17, &event1_18);
    CHECK_ERROR(err);


    // max diff: 0.00013542175292968750
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/input_block/test/test_input_block_11.npy");
    /* input_block layer[11] */
    /* input_block layer */

    /* middle_block layer */
    err = middle_block_0_res_block->forward(buffer_1280_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event1_18, &event2_0);
    CHECK_ERROR(err);

    err = middle_block_1_spatial->forward(buffer_1280_8, bufferCondition, buffer_1280_8,
                                          1, &event2_0, &event2_1);
    CHECK_ERROR(err);

    err = middle_block_2_res_block->forward(buffer_1280_8, bufferEmbed, buffer_1280_8,
                                            1, &event0_3,
                                            1, &event2_1, &event2_2);
    CHECK_ERROR(err);

    // max diff: 0.00013828277587890625
    // util::testBuffer(cmdQueue, buffer_1280_8, "unet/middle_block/test/test_middle_block.npy");
    /* middle_block layer */

    clReleaseEvent(event0_0);
    clReleaseEvent(event0_1);
    clReleaseEvent(event0_2);
    clReleaseEvent(event0_3);
    clReleaseEvent(event1_0);
    clReleaseEvent(event1_1);
    clReleaseEvent(event1_2);
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
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);
    clReleaseMemObject(bufferInput);
    clReleaseMemObject(buffer_320_64);
    clReleaseMemObject(bufferCondition);
    clReleaseMemObject(buffer_320_32);
    clReleaseMemObject(buffer_640_32);
    clReleaseMemObject(buffer_640_16);
    clReleaseMemObject(buffer_1280_16);
    clReleaseMemObject(buffer_1280_8);

    return std::vector<float>();
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
UNetModel::test(const std::vector<float> &x, long timestep, const std::vector<float> &condition) {
    cl_int err;
    cl_event event0, event1, event2, event3;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed, bufferCondition, bufferInput, buffer_1280_8;

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

    buffer_1280_8 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 4 * MODEL_CHANNELS * 8 * 8,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferTimeEmbed, CL_TRUE, 0,
                               sizeof(float) * t_emb.size(),
                               t_emb.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferCondition, CL_TRUE, 0,
                               sizeof(float) * condition.size(),
                               condition.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferInput, CL_TRUE, 0,
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

    err = output_block_1_res_block->forward(bufferInput, bufferEmbed, buffer_1280_8,
                                            1, &event2,
                                            0, nullptr,
                                            &event3);
    CHECK_ERROR(err);

    util::testBuffer(cmdQueue, buffer_1280_8, "unet/output_block/test/test_output_block_1.npy");

    // test_output_block_0 max diff: 0.00002098083496093750
    // test_output_block_1.npy max diff: 0.00002479553222656250
    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseMemObject(buffer_1280_8);
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);
    clReleaseMemObject(bufferInput);
    clReleaseMemObject(bufferCondition);
}
