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

    for (int i = 0; i < 3; i++) {
        auto folder_prefix =
                "decoder/up/0/decoder_up_0_block_" + std::to_string(i);
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
            in_channel = 256;
            out_channel = 128;
            in_skip_conv2d_weight_name = "decoder/up/0/decoder_up_0_block_0_nin_shortcut_weight.npy";
            in_skip_conv2d_bias_name = "decoder/up/0/decoder_up_0_block_0_nin_shortcut_bias.npy";
        } else {
            in_channel = 128;
            out_channel = 128;
            in_skip_conv2d_weight_name = "";
            in_skip_conv2d_bias_name = "";
        }
        up_0_res_blocks[i] = new ResBlock(context, cmdQueue, deviceId, assetManager,
                                          in_channel, 0, out_channel,
                                          in_group_norm_weight_name, in_group_norm_bias_name,
                                          in_conv2d_weight_name, in_conv2d_bias_name,
                                          "", "",
                                          out_group_norm_weight_name, out_group_norm_bias_name,
                                          out_conv2d_weight_name, out_conv2d_bias_name,
                                          in_skip_conv2d_weight_name, in_skip_conv2d_bias_name);
    }

    out_group_norm = new GroupNorm(context, cmdQueue, deviceId, assetManager,
                                   32, 128, 1e-6,
                                   "decoder/out/decoder_norm_out_weight.npy",
                                   "decoder/out/decoder_norm_out_bias.npy");

    out_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                            128, 3, 3, 1, 1,
                            "decoder/out/decoder_conv_out_weight.npy",
                            "decoder/out/decoder_conv_out_bias.npy");

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");
    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);

    post_quant_conv2d->init();
    in_conv2d->init();
    mid_res_block_1->init();
    mid_attn_block->init();
    mid_res_block_2->init();
    for (auto &block: up_3_res_blocks) {
        block->init();
    }
    up_3_up_sample->init();
    for (auto &block: up_2_res_blocks) {
        block->init();
    }
    up_2_up_sample->init();
//    for (auto &block: up_1_res_blocks) {
//        block->init();
//    }
//    up_1_up_sample->init();
//    for (auto &block: up_0_res_blocks) {
//        block->init();
//    }
//    out_group_norm->init();
//    out_conv2d->init();
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
    for (auto &block: up_0_res_blocks) {
        delete block;
    }
    delete out_group_norm;
    delete out_conv2d;
    clReleaseKernel(kernel_silu);
}

std::vector<float> Decoder::decode(const std::vector<float> &x) {
    std::vector<float> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        y[i] = 1.f / SCALE_FACTOR * x[i];
    }

    cl_int err;
    cl_event event[24];
    cl_mem bufferX, buffer_4_64, buffer_512_64, buffer_512_128, buffer_512_256, buffer_256_256, buffer_256_512, buffer_128_512, buffer_3_512;

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
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 0");
    err = post_quant_conv2d->forward(bufferX, buffer_4_64, 1, &event[0], &event[1]);
    CHECK_ERROR_THROW(err);
    delete post_quant_conv2d;
    post_quant_conv2d = nullptr;

    /* Decoder */
    buffer_512_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 512 * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR_THROW(err);

    in_conv2d->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 1");
    err = in_conv2d->forward(buffer_4_64, buffer_512_64, 1, &event[1], &event[2]);
    CHECK_ERROR_THROW(err);
    delete in_conv2d;
    in_conv2d = nullptr;

    // test_conv_in.npy max diff: 0.00000238418579101562
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_conv_in.npy");

    /* mid */
    mid_res_block_1->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 2");
    err = mid_res_block_1->forward(buffer_512_64, nullptr, buffer_512_64,
                                   0, nullptr,
                                   1, &event[2], &event[3]);
    CHECK_ERROR_THROW(err);
    delete mid_res_block_1;
    mid_res_block_1 = nullptr;

    // test_mid_block_1.npy max diff: 0.00001192092895507812
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_block_1.npy");

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 3");
    mid_attn_block->init();
    err = mid_attn_block->forward(buffer_512_64, buffer_512_64,
                                  1, &event[3], &event[4]);
    CHECK_ERROR_THROW(err);
    delete mid_attn_block;
    mid_attn_block = nullptr;

    // test_mid_attn_1.npy max diff: 0.00001525878906250000
    // util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_attn_1.npy");

    mid_res_block_2->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 4");
    err = mid_res_block_2->forward(buffer_512_64, nullptr, buffer_512_64,
                                   0, nullptr,
                                   1, &event[4], &event[5]);
    CHECK_ERROR_THROW(err);
    delete mid_res_block_2;
    mid_res_block_2 = nullptr;

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
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process %d", event_idx);
        err = block->forward(buffer_512_64, nullptr, buffer_512_64,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);
        delete block;
        block = nullptr;

        event_idx++;
    }

    up_3_up_sample->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 8");
    err = up_3_up_sample->forward(buffer_512_64, buffer_512_128,
                                  1, &event[8], &event[9]);
    CHECK_ERROR_THROW(err);
    delete up_3_up_sample;
    up_3_up_sample = nullptr;

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
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process %d", event_idx);
        err = block->forward(buffer_512_128, nullptr, buffer_512_128,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);
        delete block;
        block = nullptr;

        event_idx++;
    }

    up_2_up_sample->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 12");
    err = up_2_up_sample->forward(buffer_512_128, buffer_512_256,
                                  1, &event[12], &event[13]);
    CHECK_ERROR_THROW(err);
    delete up_2_up_sample;
    up_2_up_sample = nullptr;

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
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 13");
    err = up_1_res_blocks[0]->forward(buffer_512_256, nullptr, buffer_256_256,
                                      0, nullptr,
                                      1, &event[13], &event[14]);
    CHECK_ERROR_THROW(err);
    delete up_1_res_blocks[0];
    up_1_res_blocks[0] = nullptr;

    event_idx = 14;
    for (int i = 1; i < 3; i++) {
        auto &block = up_1_res_blocks[i];
        block->init();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process %d", event_idx);
        err = block->forward(buffer_256_256, nullptr, buffer_256_256,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);
        delete block;
        block = nullptr;

        event_idx++;
    }

    up_1_up_sample->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 16");
    err = up_1_up_sample->forward(buffer_256_256, buffer_256_512,
                                  1, &event[16], &event[17]);
    CHECK_ERROR_THROW(err);
    delete up_1_up_sample;
    up_1_up_sample = nullptr;

    // test_up_1.npy max diff: 0.00213623046875000000
    // util::testBuffer(cmdQueue, buffer_256_512, "decoder/test/test_up_1.npy");
    /* up[1] */

    /* up[0] */
    buffer_128_512 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * 128 * 512 * 512,
                                    nullptr, &err);
    CHECK_ERROR_THROW(err);

    up_0_res_blocks[0]->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 17");
    err = up_0_res_blocks[0]->forward(buffer_256_512, nullptr, buffer_128_512,
                                      0, nullptr,
                                      1, &event[17], &event[18]);
    CHECK_ERROR_THROW(err);
    delete up_0_res_blocks[0];
    up_0_res_blocks[0] = nullptr;

    event_idx = 18;
    for (int i = 1; i < 3; i++) {
        auto &block = up_0_res_blocks[i];
        block->init();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process %d", event_idx);
        err = block->forward(buffer_128_512, nullptr, buffer_128_512,
                             0, nullptr,
                             1, &event[event_idx], &event[event_idx + 1]);
        CHECK_ERROR_THROW(err);
        delete block;
        block = nullptr;

        event_idx++;
    }

    // test_up_0.npy max diff: 0.00390625000000000000
    // util::testBuffer(cmdQueue, buffer_128_512, "decoder/test/test_up_0.npy");
    /* up[0] */
    /* up */

    /* out */
    buffer_3_512 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(float) * 3 * 512 * 512,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    out_group_norm->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 20");
    err = out_group_norm->forward(buffer_128_512, buffer_128_512,
                                  1, &event[20], &event[21]);
    CHECK_ERROR_THROW(err);
    delete out_group_norm;
    out_group_norm = nullptr;

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &buffer_128_512);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &buffer_128_512);
    CHECK_ERROR_THROW(err);

    size_t outWorkSize[3] = {128 * 512 * 512};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 outWorkSize, nullptr,
                                 1, &event[21], &event[22]);
    CHECK_ERROR_THROW(err);

    out_conv2d->init();
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "process 21");
    err = out_conv2d->forward(buffer_128_512, buffer_3_512,
                              1, &event[22], &event[23]);
    CHECK_ERROR_THROW(err);
    delete out_conv2d;
    out_conv2d = nullptr;

    // test_out.npy max diff: 0.00000357627868652344
    // util::testBuffer(cmdQueue, buffer_3_512, "decoder/test/test_out.npy");
    /* out */
    /* Decoder */

    /* result */
    std::vector<float> result(3 * 512 * 512);
    err = clEnqueueReadBuffer(cmdQueue, buffer_3_512, CL_FALSE, 0,
                              sizeof(float) * result.size(), result.data(),
                              1, &event[23], nullptr);
    CHECK_ERROR_THROW(err);
    /* result */

    clFinish(cmdQueue);

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
    clReleaseMemObject(buffer_128_512);
    return result;
}

void Decoder::test(const std::vector<float> &x) {
    cl_int err;
    cl_event event[2];
    cl_mem bufferX, buffer_512_64;

    bufferX = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             sizeof(float) * x.size(),
                             nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferX, CL_FALSE, 0,
                               sizeof(float) * x.size(), x.data(),
                               0, nullptr, &event[0]);
    CHECK_ERROR_THROW(err);

    buffer_512_64 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * 512 * 64 * 64,
                                   nullptr, &err);
    CHECK_ERROR_THROW(err);

    /* test logic */
    mid_attn_block->init();
    err = mid_attn_block->forward(bufferX, buffer_512_64,
                                  1, &event[0], &event[1]);
    CHECK_ERROR_THROW(err);
    /* test logic */

    util::testBuffer(cmdQueue, buffer_512_64, "decoder/test/test_mid_attn_1.npy");

    for (auto &e: event) {
        clReleaseEvent(e);
    }
    clReleaseMemObject(bufferX);
    clReleaseMemObject(buffer_512_64);
}