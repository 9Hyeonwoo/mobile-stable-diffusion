//
// Created by 구현우 on 2023/12/07.
//

#include "GroupNorm.h"

#include <android/log.h>
#include "../util.h"

#define LOG_TAG "GROUP_NORM"

#define WORK_GROUP_SIZE 64

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

GroupNorm::GroupNorm(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t num_groups, size_t num_channels, float eps,
        const std::string &weight_name, const std::string &bias_name
) : context(context), cmdQueue(cmdQueue), num_groups(num_groups), num_channels(num_channels),
    eps(eps), weight_name(weight_name), bias_name(bias_name), event_init_weight(nullptr),
    event_init_bias(nullptr) {
    cl_int err;
    weightSize = num_channels;
    biasSize = num_channels;

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * weightSize,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(float) * biasSize,
                                nullptr, &err);
    CHECK_ERROR_THROW(err);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/group_norm.cl");
    kernel_mean = clCreateKernel(program, "local_reduction_mean", &err);
    CHECK_ERROR_THROW(err);

    kernel_var = clCreateKernel(program, "local_reduction_variance", &err);
    CHECK_ERROR_THROW(err);

    kernel_norm = clCreateKernel(program, "group_norm", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

GroupNorm::~GroupNorm() {
    clReleaseKernel(kernel_mean);
    clReleaseKernel(kernel_var);
    clReleaseKernel(kernel_norm);
    if (event_init_weight != nullptr) {
        clReleaseMemObject(bufferWeight);
        clReleaseEvent(event_init_weight);
    }
    if (event_init_bias != nullptr) {
        clReleaseMemObject(bufferBias);
        clReleaseEvent(event_init_bias);
    }
}

void GroupNorm::init() {
    if (event_init_weight != nullptr && event_init_bias != nullptr) {
        return;
    }
    cl_int err;
    auto weight = util::load_npy_file(weight_name);
    auto bias = util::load_npy_file(bias_name);

    if (weight.num_vals != bias.num_vals) {
        throw std::runtime_error("weight.shape[0] != bias.shape[0]");
    }

    if (weight.num_vals != weightSize) {
        throw std::runtime_error("weight.shape[0] != weightSize");
    }

    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_TRUE, 0,
                               sizeof(float) * weightSize,
                               weight.data<float>(), 0, nullptr, &event_init_weight);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_TRUE, 0,
                                sizeof(float) * biasSize,
                                bias.data<float>(), 0, nullptr, &event_init_bias);
    CHECK_ERROR_THROW(err);
}

cl_int GroupNorm::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event1, event2, *_event_list;

    size_t input_bytes, _num_events;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &input_bytes, nullptr);
    CHECK_ERROR(err);

    auto input_size = input_bytes / sizeof(float);
    size_t groupSize = input_size / num_groups;
    size_t reductionSize = groupSize / WORK_GROUP_SIZE;

    if (input_size % weightSize != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "input_size: %ld, weight->num_vals: %ld",
                            input_size, weightSize);
        throw std::runtime_error("input_size % weight->num_vals != 0");
    }

    if (groupSize % WORK_GROUP_SIZE != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "groupSize: %ld, WORK_GROUP_SIZE: %d",
                            groupSize, WORK_GROUP_SIZE);
        throw std::runtime_error("groupSize % WORK_GROUP_SIZE != 0");
    }

    _num_events = num_events_in_list + 2;
    _event_list = new cl_event[_num_events];
    _event_list[0] = event_init_weight;
    _event_list[1] = event_init_bias;
    for (int i = 0; i < num_events_in_list; i++) {
        _event_list[i + 2] = event_wait_list[i];
    }

    cl_mem bufferMean = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * num_groups,
                                       nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferVariance = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(float) * num_groups,
                                           nullptr, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_mean, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_mean, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_mean, 2, sizeof(float) * groupSize / reductionSize, nullptr);
    err |= clSetKernelArg(kernel_mean, 3, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    size_t globalReductionSize[1] = {num_groups * WORK_GROUP_SIZE};
    size_t localReductionSize[1] = {WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_mean, 1, nullptr, globalReductionSize,
                                 localReductionSize,
                                 _num_events, _event_list, &event1);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_var, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_var, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_var, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel_var, 3, sizeof(float) * groupSize / reductionSize, nullptr);
    err |= clSetKernelArg(kernel_var, 4, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel_var, 1, nullptr, globalReductionSize,
                                 localReductionSize, 1,
                                 &event1, &event2);
    CHECK_ERROR(err);

    size_t channelSize = input_size / num_channels;
    err = clSetKernelArg(kernel_norm, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_norm, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_norm, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel_norm, 3, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_norm, 4, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_norm, 5, sizeof(size_t), &groupSize);
    err |= clSetKernelArg(kernel_norm, 6, sizeof(size_t), &channelSize);
    err |= clSetKernelArg(kernel_norm, 7, sizeof(float), &eps);
    err |= clSetKernelArg(kernel_norm, 8, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t globalWorkSize[1] = {input_size};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_norm, 1, nullptr, globalWorkSize, nullptr, 1,
                                 &event2, event);
    CHECK_ERROR(err);

    clReleaseMemObject(bufferMean);
    clReleaseMemObject(bufferVariance);
    clReleaseEvent(event1);
    clReleaseEvent(event2);

    delete[] _event_list;

    return CL_SUCCESS;
}