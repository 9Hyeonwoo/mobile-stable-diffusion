//
// Created by 구현우 on 2023/12/10.
//

#include "FeedForward.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "FEED_FORWARD"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      return err; \
    }

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

FeedForward::FeedForward(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager, size_t dim,
        const std::string &geglu_linear_weight_name,
        const std::string &geglu_linear_bias_name,
        const std::string &net_linear_weight_name, const std::string &net_linear_bias_name
) : context(context), cmdQueue(cmdQueue) {

    geglu = new GEGLU(context, cmdQueue, deviceId, assetManager,
                      dim, dim * 4,
                      geglu_linear_weight_name, geglu_linear_bias_name);
    netLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                           dim * 4, dim,
                           net_linear_weight_name, net_linear_bias_name);
}

FeedForward::~FeedForward() {
    delete geglu;
    delete netLinear;
}

void FeedForward::init() {
    geglu->init();
    netLinear->init();
}

cl_int FeedForward::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0;
    cl_mem bufferGEGLU;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR_THROW(err);

    bufferGEGLU = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 inputBytes / geglu->weightShape[1] * geglu->weightShape[0] / 2,
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = geglu->forward(input, bufferGEGLU, num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR(err);

    // max diff: 0.00000786781311035156
    // util::testBuffer(cmdQueue, bufferGEGLU, "unet/input_block/test/test_basic_ff_geglu.npy");

    err = netLinear->forward(bufferGEGLU, output, 1, &event0, event);
    CHECK_ERROR(err);

    clReleaseEvent(event0);
    clReleaseMemObject(bufferGEGLU);

    return CL_SUCCESS;
}