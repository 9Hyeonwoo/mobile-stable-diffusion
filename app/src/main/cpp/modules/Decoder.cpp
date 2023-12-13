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
}

Decoder::~Decoder() {
    delete post_quant_conv2d;
    delete in_conv2d;
}

std::vector<float> Decoder::decode(const std::vector<float> &x) {
    std::vector<float> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        y[i] = 1.f / SCALE_FACTOR * x[i];
    }

    cl_int err;
    cl_event event[3];
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
    /* Decoder */

    for (auto &e : event) {
        clReleaseEvent(e);
    }
    clReleaseMemObject(bufferX);
    clReleaseMemObject(buffer_4_64);
    clReleaseMemObject(buffer_512_64);
    return y;
}

