//
// Created by 구현우 on 2023/11/27.
//

#include "LayerNorm.h"
#include "../util.h"
#include <android/log.h>

#define LOG_TAG "LAYER_NORM"
#define WORK_GROUP_SIZE 64

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

LayerNorm::LayerNorm(
        cl_context context,
        cl_command_queue cmdQueue,
        size_t dim,
        const std::string &weight_name, const std::string &bias_name,
        LayerNormKernel &kernel
) : context(context), cmdQueue(cmdQueue), weight_name(weight_name), bias_name(bias_name),
    bufferWeight(nullptr), bufferBias(nullptr), kernel(kernel), weightSize(dim), biasSize(dim) {
}

LayerNorm::~LayerNorm() {
    if (bufferWeight != nullptr) {
        clReleaseMemObject(bufferWeight);
    }
    if (bufferBias != nullptr) {
        clReleaseMemObject(bufferBias);
    }
}

void LayerNorm::init() {
    if (bufferWeight != nullptr && bufferBias != nullptr) {
        return;
    }
    size_t weight_num_vals, bias_num_vals;
    bufferWeight = util::load_npy_file(weight_name, &weight_num_vals, context, cmdQueue);
    bufferBias = util::load_npy_file(bias_name, &bias_num_vals, context, cmdQueue);
    if (weight_num_vals != bias_num_vals) {
        throw std::runtime_error("weightSize != biasSize");
    }

    if (weight_num_vals != weightSize) {
        throw std::runtime_error("weightSize != weight->num_vals");
    }

    if (weightSize % WORK_GROUP_SIZE != 0) {
        throw std::runtime_error("weightSize % WORK_GROUP_SIZE != 0");
    }
}

cl_int LayerNorm::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    size_t input_bytes;
    cl_event event1, event2;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &input_bytes, nullptr);
    CHECK_ERROR(err);

    auto input_size = input_bytes / sizeof(cl_float);

    if (input_size % weightSize != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "input_size: %ld, weight->num_vals: %ld",
                            input_size, weightSize);
        throw std::runtime_error("input_size % weight->num_vals != 0");
    }

    cl_mem bufferMean = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * input_size / weightSize,
                                       nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferVariance = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(float) * input_size / weightSize,
                                           nullptr, &err);
    CHECK_ERROR(err);

    size_t reductionSize = weightSize / WORK_GROUP_SIZE;
    err = clSetKernelArg(kernel.mean, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel.mean, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel.mean, 2, sizeof(float) * WORK_GROUP_SIZE, nullptr);
    err |= clSetKernelArg(kernel.mean, 3, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    size_t globalReductionSize[1] = {input_size / reductionSize};
    size_t localReductionSize[1] = {WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.mean, 1, nullptr, globalReductionSize,
                                 localReductionSize,
                                 num_events_in_list, event_wait_list, &event1);
    CHECK_ERROR(err);

//    clWaitForEvents(1, &event1);
//    util::testBuffer(cmdQueue, bufferMean, "encoder/test/local_mean_test_fp32.npy");

    err = clSetKernelArg(kernel.variance, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel.variance, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel.variance, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel.variance, 3, sizeof(float) * WORK_GROUP_SIZE, nullptr);
    err |= clSetKernelArg(kernel.variance, 4, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel.variance, 1, nullptr, globalReductionSize,
                                 localReductionSize, 1,
                                 &event1, &event2);
    CHECK_ERROR(err);

//    clWaitForEvents(1, &event2);
//    util::testBuffer(cmdQueue, bufferVariance, "encoder/test/local_var_test_fp32.npy");

    err = clSetKernelArg(kernel.normalization, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel.normalization, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel.normalization, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel.normalization, 3, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel.normalization, 4, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel.normalization, 5, sizeof(size_t), &weightSize);
    err |= clSetKernelArg(kernel.normalization, 6, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t globalWorkSize[1] = {input_size};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.normalization, 1, nullptr, globalWorkSize,
                                 nullptr, 1,
                                 &event2, event);
    CHECK_ERROR(err);

//    clWaitForEvents(1, event);
//    util::testBuffer(cmdQueue, output, "encoder/test/layer_norm_0_test_fp32.npy");

    clReleaseMemObject(bufferMean);
    clReleaseMemObject(bufferVariance);
    clReleaseEvent(event1);
    clReleaseEvent(event2);

    return CL_SUCCESS;
}