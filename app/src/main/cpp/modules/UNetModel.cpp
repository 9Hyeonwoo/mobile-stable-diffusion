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
    cl_event event1_0, event1_1, event1_2, event1_3, event1_4, event1_5, event1_6, event1_7, event1_8;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed;
    cl_mem bufferInput, bufferInputBlock_0, bufferCondition, bufferInputBlock_3;
    cl_mem bufferInputBlock_4;

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

    bufferInputBlock_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(float) * MODEL_CHANNELS * 64 * 64,
                                        nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_0_conv2d->forward(bufferInput, bufferInputBlock_0, 1, &event1_0, &event1_1);
    CHECK_ERROR(err);

    // x=seed45.npy. timestep=981. max diff: 0.00000059604644775391
    // util::testBuffer(cmdQueue, bufferInputBlock_0, "unet/input_block/test/test_input_block_0_conv2d.npy");
    /* input_block layer[0] */

    /* input_block layer[1] */

    bufferCondition = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * condition.size(),
                                     nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferCondition, CL_FALSE, 0,
                               sizeof(float) * condition.size(),
                               condition.data(), 0, nullptr, &event1_2);

    err = input_block_1_res_block->forward(bufferInputBlock_0, bufferEmbed, bufferInputBlock_0,
                                           1, &event0_3,
                                           1, &event1_1, &event1_2);
    CHECK_ERROR(err);

    err = input_block_1_spatial->forward(bufferInputBlock_0, bufferCondition, bufferInputBlock_0,
                                         2, &event1_2, &event1_3);
    CHECK_ERROR(err);
    /* input_block layer[1] */

    /* input_block layer[2] */
    err = input_block_2_res_block->forward(bufferInputBlock_0, bufferEmbed, bufferInputBlock_0,
                                           1, &event0_3,
                                           1, &event1_3, &event1_4);
    CHECK_ERROR(err);

    err = input_block_2_spatial->forward(bufferInputBlock_0, bufferCondition, bufferInputBlock_0,
                                         1, &event1_4, &event1_5);
    CHECK_ERROR(err);

    // max diff: 0.00001168251037597656
    // util::testBuffer(cmdQueue, bufferInputBlock_0, "unet/input_block/test/test_input_block_2.npy");
    /* input_block layer[2] */

    /* input_block layer[3] */
    bufferInputBlock_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(float) * MODEL_CHANNELS * 32 * 32,
                                        nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_3_conv2d->forward(bufferInputBlock_0, bufferInputBlock_3,
                                        1, &event1_5, &event1_6);
    CHECK_ERROR(err);

    // max diff: 0.00001525878906250000
    // util::testBuffer(cmdQueue, bufferInputBlock_3, "unet/input_block/test/test_input_block_3.npy");
    /* input_block layer[3] */

    /* input_block layer[4] */
    bufferInputBlock_4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(float) * 2 * MODEL_CHANNELS * 32 * 32,
                                        nullptr, &err);
    CHECK_ERROR(err);

    err = input_block_4_res_block->forward(bufferInputBlock_3, bufferEmbed, bufferInputBlock_4,
                                           1, &event0_3,
                                           1, &event1_6, &event1_7);
    CHECK_ERROR(err);

    // max diff: 0.00001382827758789062
    // util::testBuffer(cmdQueue, bufferInputBlock_4, "unet/input_block/test/test_input_block_4_res.npy");

    err = input_block_4_spatial->forward(bufferInputBlock_4, bufferCondition, bufferInputBlock_4,
                                         1, &event1_7, &event1_8);
    CHECK_ERROR(err);

    // max diff: 0.00002956390380859375
    // util::testBuffer(cmdQueue, bufferInputBlock_4, "unet/input_block/test/test_input_block_4.npy");
    /* input_block layer[4] */
    /* input_block layer */

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
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);
    clReleaseMemObject(bufferInput);
    clReleaseMemObject(bufferInputBlock_0);
    clReleaseMemObject(bufferCondition);
    clReleaseMemObject(bufferInputBlock_3);
    clReleaseMemObject(bufferInputBlock_4);

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