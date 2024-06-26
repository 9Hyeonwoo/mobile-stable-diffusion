//
// Created by 구현우 on 2023/12/11.
//

#include "UpSample.h"

#include "../util.h"

#include <android/log.h>

#define LOG_TAG "UPSAMPLE"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      return err;                     \
    }                    \

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

UpSample::UpSample(
        cl_context context, cl_command_queue cmdQueue,
        size_t in_channel, size_t out_channel, size_t kernel_size, int stride, int padding,
        const std::string &weight_name, const std::string &bias_name,
        std::shared_ptr<ConvKernel> convKernel,
        std::shared_ptr<UpSampleKernel> upSampleKernel
) : context(context), cmdQueue(cmdQueue), scale(2), kernel(upSampleKernel) {

    conv2d = new Conv2D(context, cmdQueue,
                        in_channel, out_channel, kernel_size, stride, padding,
                        weight_name, bias_name, convKernel);
}

UpSample::~UpSample() {
    delete conv2d;
}

void UpSample::init() {
    conv2d->init();
}

cl_int UpSample::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0;
    cl_mem bufferUpSample;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    size_t inputSize = inputBytes / sizeof(float);
    size_t inputChannel = conv2d->weightShape[1];

    if (inputSize % inputChannel != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] inputSize(%ld) %% weightShape[1](%ld) != 0",
                            __FILE__, __LINE__, inputSize, inputChannel);
        throw std::runtime_error("inputSize % weightShape[1] != 0");
    }

    size_t heightXwidth = inputSize / inputChannel;
    auto height = static_cast<size_t>(sqrt((heightXwidth)));

    if (height * height != heightXwidth) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] height(%ld)^2 != heightXwidth(%ld)",
                            __FILE__, __LINE__, height, heightXwidth);
        throw std::runtime_error("height^2 != heightXwidth");
    }

    size_t outputBytes;
    err = clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(size_t), &outputBytes, nullptr);
    CHECK_ERROR(err);

    size_t outputSize = outputBytes / sizeof(float);
    if (outputSize != heightXwidth * scale * scale * conv2d->weightShape[0]) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] outputSize(%ld) != (height(%ld) * scale(%ld))^2 * weightShape[0](%ld)",
                            __FILE__, __LINE__, outputSize, height, scale, conv2d->weightShape[0]);
        throw std::runtime_error("outputSize != (height * scale)^2 * weightShape[0]");
    }

    bufferUpSample = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * inputChannel * (heightXwidth * scale * scale),
                                    nullptr, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel->up_sample_nearest, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->up_sample_nearest, 1, sizeof(cl_mem), &bufferUpSample);
    err |= clSetKernelArg(kernel->up_sample_nearest, 2, sizeof(size_t), &scale);
    CHECK_ERROR(err);

    size_t upSampleGlobalSize[3] = {inputChannel, height, height};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->up_sample_nearest, 3, nullptr,
                                 upSampleGlobalSize, nullptr, num_events_in_list, event_wait_list,
                                 &event0);
    CHECK_ERROR(err);

    err = conv2d->forward(bufferUpSample, output, 1, &event0, event);
    CHECK_ERROR(err);

    clReleaseEvent(event0);
    clReleaseMemObject(bufferUpSample);

    return CL_SUCCESS;
}