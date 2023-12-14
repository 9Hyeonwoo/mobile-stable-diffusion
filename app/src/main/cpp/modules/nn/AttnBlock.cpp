//
// Created by 구현우 on 2023/12/14.
//

#include "AttnBlock.h"

#include "../util.h"

#include <android/log.h>

#define WORK_GROUP_SIZE 64

#define LOG_TAG "ATTN_BLOCK"

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

AttnBlock::AttnBlock(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t in_channels,
        const char *group_norm_weight, const char *group_norm_bias,
        const char *q_conv2d_weight_name, const char *q_conv2d_bias_name,
        const char *k_conv2d_weight_name, const char *k_conv2d_bias_name,
        const char *v_conv2d_weight_name, const char *v_conv2d_bias_name,
        const char *out_conv2d_weight_name, const char *out_conv2d_bias_name
) : context(context), cmdQueue(cmdQueue), in_channels(in_channels) {
    cl_int err;

    groupNorm = new GroupNorm(context, cmdQueue, deviceId, assetManager,
                              32, in_channels, 1e-6,
                              group_norm_weight, group_norm_bias);

    to_q_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                             in_channels, in_channels, 1, 1, 0,
                             q_conv2d_weight_name, q_conv2d_bias_name);

    to_k_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                             in_channels, in_channels, 1, 1, 0,
                             k_conv2d_weight_name, k_conv2d_bias_name);

    to_v_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                             in_channels, in_channels, 1, 1, 0,
                             v_conv2d_weight_name, v_conv2d_bias_name);

    out_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager,
                            in_channels, in_channels, 1, 1, 0,
                            out_conv2d_weight_name, out_conv2d_bias_name);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_permute3D_0_2_1 = clCreateKernel(program, "permute3D__0_2_1", &err);
    CHECK_ERROR_THROW(err);

    kernel_batch_matmul = clCreateKernel(program, "batch_matmul", &err);
    CHECK_ERROR_THROW(err);

    kernel_softmax = clCreateKernel(program, "softmax", &err);
    CHECK_ERROR_THROW(err);

    kernel_elem_add = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

AttnBlock::~AttnBlock() {
    delete groupNorm;
    delete to_q_conv2d;
    delete to_k_conv2d;
    delete to_v_conv2d;
    delete out_conv2d;
    clReleaseKernel(kernel_permute3D_0_2_1);
    clReleaseKernel(kernel_batch_matmul);
    clReleaseKernel(kernel_softmax);
    clReleaseKernel(kernel_elem_add);
}

cl_int AttnBlock::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                          const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    cl_event events[10];

    size_t inputBytes;
    cl_mem bufferNorm, bufferQ, bufferK, bufferV, bufferPermuteQ, bufferQK, bufferPermuteQK;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    size_t heightXwidth = inputBytes / sizeof(float) / in_channels;

    bufferNorm = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                inputBytes,
                                nullptr, &err);
    CHECK_ERROR(err);

    bufferQ = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             inputBytes,
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             inputBytes,
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             inputBytes,
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferPermuteQ = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    inputBytes,
                                    nullptr, &err);
    CHECK_ERROR(err);

    bufferQK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(float) * heightXwidth * heightXwidth,
                              nullptr, &err);
    CHECK_ERROR(err)

    bufferPermuteQK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(float) * heightXwidth * heightXwidth,
                                     nullptr, &err);
    CHECK_ERROR(err)

    groupNorm->init();
    err = groupNorm->forward(input, bufferNorm, num_events_in_list, event_wait_list, &events[0]);
    CHECK_ERROR(err);

    to_q_conv2d->init();
    err = to_q_conv2d->forward(bufferNorm, bufferQ, 1, &events[0], &events[1]);
    CHECK_ERROR(err);

    to_k_conv2d->init();
    err = to_k_conv2d->forward(bufferNorm, bufferK, 1, &events[0], &events[2]);
    CHECK_ERROR(err);

    to_v_conv2d->init();
    err = to_v_conv2d->forward(bufferNorm, bufferV, 1, &events[0], &events[7]);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_permute3D_0_2_1, 0, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(kernel_permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermuteQ);
    CHECK_ERROR(err);

    size_t global_size[3] = {1, in_channels, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_0_2_1, 3, nullptr,
                                 global_size, nullptr, 1, &events[1], &events[3]);
    CHECK_ERROR(err);

    float scale = 1.f / sqrtf(static_cast<float>(in_channels));
    err = clSetKernelArg(kernel_batch_matmul, 0, sizeof(cl_mem), &bufferPermuteQ);
    err |= clSetKernelArg(kernel_batch_matmul, 1, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(kernel_batch_matmul, 2, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(kernel_batch_matmul, 3, sizeof(size_t), &in_channels);
    err |= clSetKernelArg(kernel_batch_matmul, 4, sizeof(float), &scale);
    CHECK_ERROR(err);

    size_t QKGlobalSize[3] = {1, heightXwidth, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_batch_matmul, 3, nullptr,
                                 QKGlobalSize, nullptr, 2, &events[2], &events[4]);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(kernel_softmax, 1, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(kernel_softmax, 2, sizeof(float) * WORK_GROUP_SIZE, nullptr);
    err |= clSetKernelArg(kernel_softmax, 3, sizeof(float) * heightXwidth, nullptr);
    err |= clSetKernelArg(kernel_softmax, 4, sizeof(size_t), &heightXwidth);
    CHECK_ERROR(err);

    size_t softmaxGlobalSize[1] = {heightXwidth * WORK_GROUP_SIZE};
    size_t softmaxLocalSize[1] = {WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_softmax, 1, nullptr,
                                 softmaxGlobalSize, softmaxLocalSize, 1, &events[4], &events[5]);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_permute3D_0_2_1, 0, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(kernel_permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermuteQK);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_0_2_1, 3, nullptr,
                                 QKGlobalSize, nullptr, 1, &events[5], &events[6]);
    CHECK_ERROR(err);

    float identity = 1.f;
    err = clSetKernelArg(kernel_batch_matmul, 0, sizeof(cl_mem), &bufferV);
    err |= clSetKernelArg(kernel_batch_matmul, 1, sizeof(cl_mem), &bufferPermuteQK);
    err |= clSetKernelArg(kernel_batch_matmul, 2, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(kernel_batch_matmul, 3, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(kernel_batch_matmul, 4, sizeof(float), &identity);
    CHECK_ERROR(err);

    size_t VQKGlobalSize[3] = {1, in_channels, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_batch_matmul, 3, nullptr,
                                 VQKGlobalSize, nullptr, 2, &events[6], &events[8]);
    CHECK_ERROR(err);

    out_conv2d->init();
    err = out_conv2d->forward(bufferQ, bufferK, 1, &events[8], &events[9]);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_elem_add, 0, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(kernel_elem_add, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_elem_add, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t elemAddGlobalSize[1] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_elem_add, 1, nullptr,
                                 elemAddGlobalSize, nullptr, 1, &events[9], event);
    CHECK_ERROR(err);

    clReleaseMemObject(bufferNorm);
    clReleaseMemObject(bufferQ);
    clReleaseMemObject(bufferK);
    clReleaseMemObject(bufferV);
    clReleaseMemObject(bufferPermuteQ);
    clReleaseMemObject(bufferQK);
    clReleaseMemObject(bufferPermuteQK);
    for (auto &e: events) {
        clReleaseEvent(e);
    }

    return CL_SUCCESS;
}