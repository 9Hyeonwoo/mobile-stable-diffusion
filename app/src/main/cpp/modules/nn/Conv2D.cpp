//
// Created by 구현우 on 2023/12/07.
//

#include "Conv2D.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "CONV2D"

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

Conv2D::Conv2D(
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t in_channel, size_t out_channel, size_t kernel_size, int stride, int padding,
        const char *weight_name,
        const char *bias_name
) : cmdQueue(cmdQueue), stride(stride), padding(padding), weight_name(weight_name),
    bias_name(bias_name), event_init_weight(nullptr), event_init_bias(nullptr) {
    cl_int err;

    weightShape = std::vector<size_t>({out_channel, in_channel, kernel_size, kernel_size});
    biasShape = std::vector<size_t>({out_channel});

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * out_channel * in_channel * kernel_size *
                                  kernel_size,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(float) * out_channel,
                                nullptr, &err);
    CHECK_ERROR_THROW(err);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/conv2d.cl");

    kernel = clCreateKernel(program, "conv2d", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);

    clSetMemObjectDestructorCallback(bufferWeight, [](cl_mem memobj, void *user_data) {
        __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "Conv2D release memobj weight");
    }, nullptr);
}

Conv2D::~Conv2D() {
    clReleaseMemObject(bufferWeight);
    clReleaseMemObject(bufferBias);
    clReleaseKernel(kernel);
    if (event_init_weight != nullptr) {
        clReleaseEvent(event_init_weight);
    }
    if (event_init_bias != nullptr) {
        clReleaseEvent(event_init_bias);
    }
}

void Conv2D::init() {
    cl_int err;
    auto weight = util::load_npy_file(weight_name);
    auto bias = util::load_npy_file(bias_name);

    if (weight.num_vals != (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D weight file size != constructor weightShape");
        throw std::runtime_error("Conv2D weight file size != constructor weightShape");
    }

    if (bias.num_vals != biasShape[0]) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D bias file size != constructor biasShape");
        throw std::runtime_error("Conv2D bias file size != constructor biasShape");
    }

    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_FALSE, 0,
                               sizeof(float) * weight.num_vals,
                               weight.data<float>(), 0, nullptr, &event_init_weight);
    err |= clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_FALSE, 0,
                                sizeof(float) * bias.num_vals,
                                bias.data<float>(), 0, nullptr, &event_init_bias);
    CHECK_ERROR_THROW(err);
}

/*
 * Assume square shaped `input` where height = width.
 */
cl_int Conv2D::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    auto _num_event_list = num_events_in_list + 2;
    auto *event_list = new cl_event[_num_event_list];
    event_list[0] = event_init_weight;
    event_list[1] = event_init_bias;
    for (int i = 0; i < num_events_in_list; i++) {
        event_list[i + 2] = event_wait_list[i];
    }

    if (input == output) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Conv2D not support input == output");
        throw std::runtime_error("Conv2D not support input == output");
    }

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    auto inputSize = inputBytes / sizeof(float);
    inputSize /= weightShape[1];
    inputSize = static_cast<size_t>(sqrt(static_cast<float>(inputSize)));

    auto outputSize = getOutputSize(inputSize);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel, 5, sizeof(size_t), &weightShape[1]);
    err |= clSetKernelArg(kernel, 6, sizeof(size_t), &weightShape[2]);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &padding);
    CHECK_ERROR(err);

    size_t globalSize[3] = {weightShape[0], outputSize, outputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, nullptr, globalSize, nullptr,
                                 _num_event_list, event_list, event);
    CHECK_ERROR(err);

    delete[] event_list;

    return CL_SUCCESS;
}

size_t Conv2D::getOutputSize(size_t inputSize) {
    return (inputSize + 2 * padding - weightShape[2]) / stride + 1;
}