//
// Created by 구현우 on 2023/12/14.
//

#include "Decoder.h"

#include <android/log.h>
#include "util.h"

#define LOG_TAG "DECODER"

#define SCALE_FACTOR 0.18215f

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) {   \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

Decoder::Decoder(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager
) : context(context), cmdQueue(cmdQueue), deviceId(deviceId) {
    cl_int err;

    post_quant_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                                   4, 4, 1, 1, 0,
                                   "decoder/post_quant_conv_weight.npy",
                                   "decoder/post_quant_conv_bias.npy");

    in_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                           4, 512, 3, 1, 1,
                           "decoder/decoder_conv_in_weight.npy",
                           "decoder/decoder_conv_in_bias.npy");

    mid_res_block_1 = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                   512, 0, 512,
                                   "decoder/mid/decoder_mid_block_1_norm1_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_norm1_bias.npy",
                                   "decoder/mid/decoder_mid_block_1_conv1_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_conv1_bias.npy",
                                   nullptr, nullptr,
                                   "decoder/mid/decoder_mid_block_1_norm2_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_norm2_bias.npy",
                                   "decoder/mid/decoder_mid_block_1_conv2_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_conv2_bias.npy",
                                   nullptr, nullptr);

    mid_attn_block = new AttnBlock(context, cmdQueue, deviceId, assetManager,
                                   512,
                                   "decoder/mid/decoder_mid_attn_1_norm_weight.npy",
                                   "decoder/mid/decoder_mid_attn_1_norm_bias.npy",
                                   "decoder/mid/decoder_mid_attn_1_q_weight.npy",
                                   "decoder/mid/decoder_mid_attn_1_q_bias.npy",
                                   "decoder/mid/decoder_mid_attn_1_k_weight.npy",
                                   "decoder/mid/decoder_mid_attn_1_k_bias.npy",
                                   "decoder/mid/decoder_mid_attn_1_v_weight.npy",
                                   "decoder/mid/decoder_mid_attn_1_v_bias.npy",
                                   "decoder/mid/decoder_mid_attn_1_proj_out_weight.npy",
                                   "decoder/mid/decoder_mid_attn_1_proj_out_bias.npy");
}

Decoder::~Decoder() {
    delete post_quant_conv2d;
    delete in_conv2d;
    delete mid_res_block_1;
    delete mid_attn_block;
}

std::vector<float> Decoder::decode(const std::vector<float> &x) {
    std::vector<float> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        y[i] = 1.f / SCALE_FACTOR * x[i];
    }

    cl_int err;
    cl_event event[5];
    cl_mem bufferX, buffer_4_64, buffer_512_64;

    bufferX = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             sizeof(float) * x.size(),
                             nullptr, &err);
    CHECK_ERROR_THROW(err);

    buffer_4_64 = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * 4 * 64 * 64,
                                 nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferX, CL_FALSE, 0,
                               sizeof(float) * x.size(), y.data(),
                               0, nullptr, &event[0]);
    CHECK_ERROR_THROW(err);

    post_quant_conv2d->init();
    err = post_quant_conv2d->forward(bufferX, buffer_4_64, 1, &event[0], &event[1]);
    CHECK_ERROR_THROW(err);

    /* Decoder */
    buffer_512_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 512 * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR_THROW(err);

    in_conv2d->init();
    err = in_conv2d->forward(buffer_4_64, buffer_512_64, 1, &event[1], &event[2]);
    CHECK_ERROR_THROW(err);

    // test_conv_in.npy max diff: 0.00000238418579101562
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_conv_in.npy");

    mid_res_block_1->init();
    err = mid_res_block_1->forward(buffer_512_64, nullptr, buffer_512_64,
                                   0, nullptr,
                                   1, &event[2], &event[3]);
    CHECK_ERROR_THROW(err);

    // test_mid_block_1.npy max diff: 0.00001192092895507812
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_block_1.npy");

    err = mid_attn_block->forward(buffer_512_64, buffer_512_64,
                                  1, &event[3], &event[4]);
    CHECK_ERROR_THROW(err);

    // test_mid_attn_1.npy max diff: 0.00001525878906250000
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_attn_1.npy");
    /* Decoder */

    for (auto &e: event) {
        clReleaseEvent(e);
    }
    clReleaseMemObject(bufferX);
    clReleaseMemObject(buffer_4_64);
    clReleaseMemObject(buffer_512_64);
    return y;
}

