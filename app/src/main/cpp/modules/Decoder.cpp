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
) : context(context), cmdQueue(cmdQueue) {
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
                                   "", "",
                                   "decoder/mid/decoder_mid_block_1_norm2_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_norm2_bias.npy",
                                   "decoder/mid/decoder_mid_block_1_conv2_weight.npy",
                                   "decoder/mid/decoder_mid_block_1_conv2_bias.npy",
                                   "", "");

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

    mid_res_block_2 = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                   512, 0, 512,
                                   "decoder/mid/decoder_mid_block_2_norm1_weight.npy",
                                   "decoder/mid/decoder_mid_block_2_norm1_bias.npy",
                                   "decoder/mid/decoder_mid_block_2_conv1_weight.npy",
                                   "decoder/mid/decoder_mid_block_2_conv1_bias.npy",
                                   "", "",
                                   "decoder/mid/decoder_mid_block_2_norm2_weight.npy",
                                   "decoder/mid/decoder_mid_block_2_norm2_bias.npy",
                                   "decoder/mid/decoder_mid_block_2_conv2_weight.npy",
                                   "decoder/mid/decoder_mid_block_2_conv2_bias.npy",
                                   "", "");

    for (int i = 0; i < 3; i++) {
        auto folder_prefix =
                "decoder/up/3/decoder_up_3_block_" + std::to_string(i);
        auto in_group_norm_weight_name = folder_prefix + "_norm1_weight.npy";
        auto in_group_norm_bias_name = folder_prefix + "_norm1_bias.npy";
        auto in_conv2d_weight_name = folder_prefix + "_conv1_weight.npy";
        auto in_conv2d_bias_name = folder_prefix + "_conv1_bias.npy";
        auto out_group_norm_weight_name = folder_prefix + "_norm2_weight.npy";
        auto out_group_norm_bias_name = folder_prefix + "_norm2_bias.npy";
        auto out_conv2d_weight_name = folder_prefix + "_conv2_weight.npy";
        auto out_conv2d_bias_name = folder_prefix + "_conv2_bias.npy";
        up_3_res_blocks[i] = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                          512, 0, 512,
                                          in_group_norm_weight_name,
                                          in_group_norm_bias_name,
                                          in_conv2d_weight_name,
                                          in_conv2d_bias_name,
                                          "", "",
                                          out_group_norm_weight_name,
                                          out_group_norm_bias_name,
                                          out_conv2d_weight_name,
                                          out_conv2d_bias_name,
                                          "", "");
    }

    up_3_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                  512, 512, 3, 1, 1,
                                  "decoder/up/3/decoder_up_3_upsample_conv_weight.npy",
                                  "decoder/up/3/decoder_up_3_upsample_conv_bias.npy");

    for (int i = 0; i < 3; i++) {
        auto folder_prefix =
                "decoder/up/2/decoder_up_2_block_" + std::to_string(i);
        auto in_group_norm_weight_name = folder_prefix + "_norm1_weight.npy";
        auto in_group_norm_bias_name = folder_prefix + "_norm1_bias.npy";
        auto in_conv2d_weight_name = folder_prefix + "_conv1_weight.npy";
        auto in_conv2d_bias_name = folder_prefix + "_conv1_bias.npy";
        auto out_group_norm_weight_name = folder_prefix + "_norm2_weight.npy";
        auto out_group_norm_bias_name = folder_prefix + "_norm2_bias.npy";
        auto out_conv2d_weight_name = folder_prefix + "_conv2_weight.npy";
        auto out_conv2d_bias_name = folder_prefix + "_conv2_bias.npy";
        up_2_res_blocks[i] = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                          512, 0, 512,
                                          in_group_norm_weight_name,
                                          in_group_norm_bias_name,
                                          in_conv2d_weight_name,
                                          in_conv2d_bias_name,
                                          "", "",
                                          out_group_norm_weight_name,
                                          out_group_norm_bias_name,
                                          out_conv2d_weight_name,
                                          out_conv2d_bias_name,
                                          "", "");
    }

    up_2_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                  512, 512, 3, 1, 1,
                                  "decoder/up/2/decoder_up_2_upsample_conv_weight.npy",
                                  "decoder/up/2/decoder_up_2_upsample_conv_bias.npy");

    for (int i = 0; i < 3; i++) {
        auto folder_prefix =
                "decoder/up/1/decoder_up_1_block_" + std::to_string(i);
        auto in_group_norm_weight_name = folder_prefix + "_norm1_weight.npy";
        auto in_group_norm_bias_name = folder_prefix + "_norm1_bias.npy";
        auto in_conv2d_weight_name = folder_prefix + "_conv1_weight.npy";
        auto in_conv2d_bias_name = folder_prefix + "_conv1_bias.npy";
        auto out_group_norm_weight_name = folder_prefix + "_norm2_weight.npy";
        auto out_group_norm_bias_name = folder_prefix + "_norm2_bias.npy";
        auto out_conv2d_weight_name = folder_prefix + "_conv2_weight.npy";
        auto out_conv2d_bias_name = folder_prefix + "_conv2_bias.npy";
        size_t in_channel, out_channel;
        std::string in_skip_conv2d_weight_name, in_skip_conv2d_bias_name;
        if (i == 0) {
            in_channel = 512;
            out_channel = 256;
            in_skip_conv2d_weight_name = "decoder/up/1/decoder_up_1_block_0_nin_shortcut_weight.npy";
            in_skip_conv2d_bias_name = "decoder/up/1/decoder_up_1_block_0_nin_shortcut_bias.npy";
        } else {
            in_channel = 256;
            out_channel = 256;
            in_skip_conv2d_weight_name = "";
            in_skip_conv2d_bias_name = "";
        }
        up_1_res_blocks[i] = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                          in_channel, 0, out_channel,
                                          in_group_norm_weight_name, in_group_norm_bias_name,
                                          in_conv2d_weight_name, in_conv2d_bias_name,
                                          "", "",
                                          out_group_norm_weight_name, out_group_norm_bias_name,
                                          out_conv2d_weight_name, out_conv2d_bias_name,
                                          in_skip_conv2d_weight_name, in_skip_conv2d_bias_name);
    }

    up_1_up_sample = new UpSample(context, cmdQueue, deviceId, assetManager,
                                  256, 256, 3, 1, 1,
                                  "decoder/up/1/decoder_up_1_upsample_conv_weight.npy",
                                  "decoder/up/1/decoder_up_1_upsample_conv_bias.npy");
}

Decoder::~Decoder() {
    delete post_quant_conv2d;
    delete in_conv2d;
    delete mid_res_block_1;
    delete mid_attn_block;
    delete mid_res_block_2;
    for (auto &block: up_3_res_blocks) {
        delete block;
    }
    delete up_3_up_sample;
    for (auto &block: up_2_res_blocks) {
        delete block;
    }
    delete up_2_up_sample;
    for (auto &block: up_1_res_blocks) {
        delete block;
    }
    delete up_1_up_sample;
}

std::vector<float> Decoder::decode(const std::vector<float> &x) {
    std::vector<float> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        y[i] = 1.f / SCALE_FACTOR * x[i];
    }

    cl_int err;
    cl_event event[18];
    cl_mem bufferX, buffer_4_64, buffer_512_64, buffer_512_128, buffer_512_256, buffer_256_256, buffer_256_512;

    bufferX = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             sizeof(float) * x.size(),
                             nullptr, &err);
    CHECK_ERROR_THROW(err);

    buffer_4_64 = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * 4 * 64 * 64,
                                 nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferX, CL_FALSE, 0,
                               sizeof(float) * y.size(), y.data(),
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

    /* mid */
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

    mid_res_block_2->init();
    err = mid_res_block_2->forward(buffer_512_64, nullptr, buffer_512_64,
                                   0, nullptr,
                                   1, &event[4], &event[5]);
    CHECK_ERROR_THROW(err);

    // test_mid_block_2.npy max diff: 0.00003004074096679688
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_block_2.npy");
    /* mid */

    /* up */
    /* up[3] */
    buffer_512_128 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 512 * 128 * 128,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    int event_idx = 5;
    for (auto &block: up_3_res_blocks) {
        block->init();
        err = block->forward(buffer_512_64, nullptr, buffer_512_64,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);

        event_idx++;
    }

    up_3_up_sample->init();
    err = up_3_up_sample->forward(buffer_512_64, buffer_512_128,
                                  1, &event[8], &event[9]);
    CHECK_ERROR_THROW(err);

    // test_up_3.npy max diff: 0.00012969970703125000
    // util::testBuffer(cmdQueue, buffer_512_128, "decoder/test/test_up_3.npy");
    /* up[3] */

    /* up[2] */
    buffer_512_256 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 512 * 256 * 256,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    event_idx = 9;
    for (auto &block: up_2_res_blocks) {
        block->init();
        err = block->forward(buffer_512_128, nullptr, buffer_512_128,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);

        event_idx++;
    }

    up_2_up_sample->init();
    err = up_2_up_sample->forward(buffer_512_128, buffer_512_256,
                                  1, &event[12], &event[13]);
    CHECK_ERROR_THROW(err);

    // test_up_2.npy max diff: 0.00060272216796875000
    // util::testBuffer(cmdQueue, buffer_512_256, "decoder/test/test_up_2.npy");
    /* up[2] */

    /* up[1] */
    buffer_256_256 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 256 * 256 * 256,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    buffer_256_512 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 256 * 512 * 512,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    up_1_res_blocks[0]->init();
    err = up_1_res_blocks[0]->forward(buffer_512_256, nullptr, buffer_256_256,
                                      0, nullptr,
                                      1, &event[13], &event[14]);
    CHECK_ERROR_THROW(err);

    event_idx = 14;
    for (int i = 1; i < 3; i++) {
        auto block = up_1_res_blocks[i];
        block->init();
        err = block->forward(buffer_256_256, nullptr, buffer_256_256,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);

        event_idx++;
    }

    up_1_up_sample->init();
    err = up_1_up_sample->forward(buffer_256_256, buffer_256_512,
                                  1, &event[16], &event[17]);
    CHECK_ERROR_THROW(err);

    // test_up_1.npy max diff: 0.00213623046875000000
    // util::testBuffer(cmdQueue, buffer_256_512, "decoder/test/test_up_1.npy");
    /* up[1] */
    /* up */
    /* Decoder */

    for (auto &e: event) {
        clReleaseEvent(e);
    }
    clReleaseMemObject(bufferX);
    clReleaseMemObject(buffer_4_64);
    clReleaseMemObject(buffer_512_64);
    clReleaseMemObject(buffer_512_128);
    clReleaseMemObject(buffer_512_256);
    clReleaseMemObject(buffer_256_256);
    clReleaseMemObject(buffer_256_512);
    return y;
}

void Decoder::test(const std::vector<float> &x) {
    cl_int err;
    cl_event event[5];
    cl_mem bufferX, buffer_256_256, buffer_256_512;

    bufferX = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             sizeof(float) * x.size(),
                             nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferX, CL_FALSE, 0,
                               sizeof(float) * x.size(), x.data(),
                               0, nullptr, &event[0]);
    CHECK_ERROR_THROW(err);

    buffer_256_256 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 256 * 256 * 256,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    buffer_256_512 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 256 * 512 * 512,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    up_1_res_blocks[0]->init();
    err = up_1_res_blocks[0]->forward(bufferX, nullptr, buffer_256_256,
                                      0, nullptr,
                                      1, &event[0], &event[1]);
    CHECK_ERROR_THROW(err);

    int event_idx = 1;
    for (int i = 1; i < 3; i++) {
        auto block = up_1_res_blocks[i];
        block->init();
        err = block->forward(buffer_256_256, nullptr, buffer_256_256,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);

        event_idx++;
    }

    up_1_up_sample->init();
    err = up_1_up_sample->forward(buffer_256_256, buffer_256_512,
                                  1, &event[3], &event[4]);
    CHECK_ERROR_THROW(err);

    util::testBuffer(cmdQueue, buffer_256_512, "decoder/test/test_up_1.npy");

    for (auto &e: event) {
        clReleaseEvent(e);
    }
    clReleaseMemObject(bufferX);
    clReleaseMemObject(buffer_256_256);
    clReleaseMemObject(buffer_256_512);
}