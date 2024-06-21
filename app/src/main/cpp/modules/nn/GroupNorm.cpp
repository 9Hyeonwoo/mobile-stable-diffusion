//
// Created by 구현우 on 2023/12/07.
//

#include "GroupNorm.h"

#include <android/log.h>
#include "../util.h"

#define DEBUG 0
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

static int count = 0;

GroupNorm::GroupNorm(
        cl_context context, cl_command_queue cmdQueue,
        size_t num_groups, size_t num_channels, float eps,
        const std::string &weight_name, const std::string &bias_name,
        std::shared_ptr<GroupNormKernel> kernel
) : context(context), cmdQueue(cmdQueue), num_groups(num_groups), num_channels(num_channels),
    eps(eps), weight_name(weight_name), bias_name(bias_name), bufferWeight(nullptr),
    bufferBias(nullptr), kernel(kernel) {
    cl_int err;
    weightSize = num_channels;
    biasSize = num_channels;
}

GroupNorm::~GroupNorm() {
    if (bufferWeight != nullptr) {
        clReleaseMemObject(bufferWeight);
    }
    if (bufferBias != nullptr) {
        clReleaseMemObject(bufferBias);
    }
}

void GroupNorm::init() {
    if (bufferWeight != nullptr && bufferBias != nullptr) {
        return;
    }
    size_t weight_num_vals, bias_num_vals;
    bufferWeight = util::load_npy_file(weight_name, &weight_num_vals, context, cmdQueue);
    bufferBias = util::load_npy_file(bias_name, &bias_num_vals, context, cmdQueue);

    if (weight_num_vals != bias_num_vals) {
        throw std::runtime_error("weight.shape[0] != bias.shape[0]");
    }

    if (weight_num_vals != weightSize) {
        throw std::runtime_error("weight.shape[0] != weightSize");
    }
}

cl_int GroupNorm::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event1, event2;

    size_t input_bytes;
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

    cl_mem bufferMean = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * num_groups,
                                       nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferVariance = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(float) * num_groups,
                                           nullptr, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel->local_reduction_mean, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->local_reduction_mean, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel->local_reduction_mean, 2, sizeof(float) * groupSize / reductionSize, nullptr);
    err |= clSetKernelArg(kernel->local_reduction_mean, 3, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    size_t globalReductionSize[1] = {num_groups * WORK_GROUP_SIZE};
    size_t localReductionSize[1] = {WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->local_reduction_mean, 1, nullptr, globalReductionSize,
                                 localReductionSize,
                                 num_events_in_list, event_wait_list, &event1);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel->local_reduction_variance, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->local_reduction_variance, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel->local_reduction_variance, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel->local_reduction_variance, 3, sizeof(float) * groupSize / reductionSize, nullptr);
    err |= clSetKernelArg(kernel->local_reduction_variance, 4, sizeof(size_t), &reductionSize);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel->local_reduction_variance, 1, nullptr, globalReductionSize,
                                 localReductionSize, 1,
                                 &event1, &event2);
    CHECK_ERROR(err);

    size_t channelSize = input_size / num_channels;
    err = clSetKernelArg(kernel->group_norm, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->group_norm, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel->group_norm, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel->group_norm, 3, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->group_norm, 4, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->group_norm, 5, sizeof(size_t), &groupSize);
    err |= clSetKernelArg(kernel->group_norm, 6, sizeof(size_t), &channelSize);
    err |= clSetKernelArg(kernel->group_norm, 7, sizeof(float), &eps);
    err |= clSetKernelArg(kernel->group_norm, 8, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t globalWorkSize[1] = {input_size};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->group_norm, 1, nullptr, globalWorkSize, nullptr, 1,
                                 &event2, event);
    CHECK_ERROR(err);

#if DEBUG
    clWaitForEvents(1, event);
    if (count == 0)
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "try, component, index, input size, group size, num group, weight size, workgroup size, kernel, time(ms)\n");
    auto message =
            "0, GroupNorm, " +
            std::to_string(count++) + ", " +
            std::to_string(input_size) + ", " +
            std::to_string(groupSize) + ", " +
            std::to_string(num_groups) + ", " +
            std::to_string(weightSize) + ", " +
            std::to_string(WORK_GROUP_SIZE);
    util::printEventTime(message + ", local_reduction_mean", event1);
    util::printEventTime(message + ", local_reduction_variance", event2);
    util::printEventTime(message + ", group_norm", *event);
#endif

    clReleaseMemObject(bufferMean);
    clReleaseMemObject(bufferVariance);
    clReleaseEvent(event1);
    clReleaseEvent(event2);

    return CL_SUCCESS;
}