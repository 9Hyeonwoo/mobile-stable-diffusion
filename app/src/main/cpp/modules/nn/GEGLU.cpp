//
// Created by 구현우 on 2023/12/10.
//

#include "GEGLU.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "GEGLU"

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

GEGLU::GEGLU(
        cl_context context, cl_command_queue cmdQueue,
        size_t in_features, size_t out_features,
        const std::string &linear_weight_name, const std::string &linear_bias_name,
        std::shared_ptr<LinearKernel> linearKernel,
        std::shared_ptr<GEGLUKernel> gegluKernel
) : context(context), cmdQueue(cmdQueue), kernel(gegluKernel) {

    linear = new Linear(context, cmdQueue,
                        in_features, out_features * 2,
                        linear_weight_name, linear_bias_name, linearKernel);

    weightShape = linear->weightShape;
}

GEGLU::~GEGLU() {
    delete linear;
}

void GEGLU::init() {
    linear->init();
}

cl_int GEGLU::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0;
    cl_mem bufferLinear;

    size_t inputBytes, bufferSize;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR_THROW(err);

    bufferSize = inputBytes / sizeof(float) / linear->weightShape[1] * linear->weightShape[0];
    bufferLinear = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(float) * bufferSize,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = linear->forward(input, bufferLinear, num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR_THROW(err);

    // max diff: 0.00001072883605957031
    // util::testBuffer(cmdQueue, bufferLinear, "unet/input_block/test/test_basic_ff_geglu_proj.npy");

    err = clSetKernelArg(kernel->gelu_multiply, 0, sizeof(cl_mem), &bufferLinear);
    err = clSetKernelArg(kernel->gelu_multiply, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t globalSize[2] = {bufferSize / linear->weightShape[0], linear->weightShape[0] / 2};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->gelu_multiply, 2, nullptr, globalSize, nullptr,
                                 1, &event0, event);
    CHECK_ERROR(err);

    clReleaseEvent(event0);
    clReleaseMemObject(bufferLinear);

    return CL_SUCCESS;
}