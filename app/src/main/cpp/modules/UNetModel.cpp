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

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR(err);

    clReleaseProgram(program);
}

UNetModel::~UNetModel() {
    delete time_embed_0;
    delete time_embed_2;
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
    cl_event event0, event1, event2, event3;
    cl_mem bufferTimeEmbed, bufferEmbedTemp, bufferEmbed;

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
                               t_emb.data(), 0, nullptr, &event0);
    CHECK_ERROR(err);

    err = time_embed_0->forward(bufferTimeEmbed, bufferEmbedTemp, 1, &event0, &event1);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferEmbedTemp);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferEmbedTemp);
    CHECK_ERROR(err);

    size_t embedWorkSize[1] = {MODEL_CHANNELS * 4};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr,
                                 embedWorkSize, nullptr, 1, &event1, &event2);
    CHECK_ERROR(err);

    err = time_embed_2->forward(bufferEmbedTemp, bufferEmbed, 1, &event2, &event3);
    CHECK_ERROR(err);

    // timestep=981. max diff: 0.00000476837158203125
    // util::testBuffer(cmdQueue, bufferEmbed, "unet/test/test_time_embed.npy");

    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseMemObject(bufferTimeEmbed);
    clReleaseMemObject(bufferEmbedTemp);

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
    // util::testBuffer(embedding, "unet/test/test_timestep_embedding.npy");
    return embedding;
}