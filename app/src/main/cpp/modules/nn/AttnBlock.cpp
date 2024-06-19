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
        const std::string &group_norm_weight, const std::string &group_norm_bias,
        const std::string &q_conv2d_weight_name, const std::string &q_conv2d_bias_name,
        const std::string &k_conv2d_weight_name, const std::string &k_conv2d_bias_name,
        const std::string &v_conv2d_weight_name, const std::string &v_conv2d_bias_name,
        const std::string &out_conv2d_weight_name, const std::string &out_conv2d_bias_name,
        std::shared_ptr<ConvKernel> convKernel,
        std::shared_ptr<UtilKernel> utilKernel
) : context(context), cmdQueue(cmdQueue), in_channels(in_channels), utilKernel(utilKernel) {

    groupNorm = new GroupNorm(context, cmdQueue, deviceId, assetManager,
                              32, in_channels, 1e-6,
                              group_norm_weight, group_norm_bias);

    to_q_conv2d = new Conv2D(context, cmdQueue,
                             in_channels, in_channels, 1, 1, 0,
                             q_conv2d_weight_name, q_conv2d_bias_name, convKernel);

    to_k_conv2d = new Conv2D(context, cmdQueue,
                             in_channels, in_channels, 1, 1, 0,
                             k_conv2d_weight_name, k_conv2d_bias_name, convKernel);

    to_v_conv2d = new Conv2D(context, cmdQueue,
                             in_channels, in_channels, 1, 1, 0,
                             v_conv2d_weight_name, v_conv2d_bias_name, convKernel);

    out_conv2d = new Conv2D(context, cmdQueue,
                            in_channels, in_channels, 1, 1, 0,
                            out_conv2d_weight_name, out_conv2d_bias_name, convKernel);
}

AttnBlock::~AttnBlock() {
    delete groupNorm;
    delete to_q_conv2d;
    delete to_k_conv2d;
    delete to_v_conv2d;
    delete out_conv2d;
}

void AttnBlock::init() {
    groupNorm->init();
    to_q_conv2d->init();
    to_k_conv2d->init();
    to_v_conv2d->init();
    out_conv2d->init();
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

    err = clSetKernelArg(utilKernel->permute3D_0_2_1, 0, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(utilKernel->permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermuteQ);
    CHECK_ERROR(err);

    size_t global_size[3] = {1, in_channels, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_0_2_1, 3, nullptr,
                                 global_size, nullptr, 1, &events[1], &events[3]);
    CHECK_ERROR(err);

    float scale = 1.f / sqrtf(static_cast<float>(in_channels));
    /* naive - batch matmul
    err = clSetKernelArg(utilKernel->batch_matmul, 0, sizeof(cl_mem), &bufferPermuteQ);
    err |= clSetKernelArg(utilKernel->batch_matmul, 1, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(utilKernel->batch_matmul, 2, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(utilKernel->batch_matmul, 3, sizeof(size_t), &in_channels);
    err |= clSetKernelArg(utilKernel->batch_matmul, 4, sizeof(float), &scale);
    CHECK_ERROR(err);

    size_t QKGlobalSize[3] = {1, heightXwidth, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->batch_matmul, 3, nullptr,
                                 QKGlobalSize, nullptr, 2, &events[2], &events[4]);
    CHECK_ERROR(err);
    naive - batch matmul */

    size_t tile_size = 128, reg_size = 8, tile_size_k = 16;
    /* optimized batch matmul - Q x K */
    if (heightXwidth % tile_size != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] heightXwidth(%ld) %% tile_size(%ld) != 0\n", __FILE__,
                            __LINE__, heightXwidth, tile_size);
        return CL_INVALID_VALUE;
    }
    if (in_channels % tile_size_k != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] in_channels(%ld) %% tile_size(16) != 0\n", __FILE__,
                            __LINE__, in_channels);
        return CL_INVALID_VALUE;
    }
    err = clSetKernelArg(utilKernel->batch_matmul_scale, 0, sizeof(cl_mem), &bufferPermuteQ);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 1, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 2, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 3, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 4, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 5, sizeof(size_t), &in_channels);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 6, sizeof(float), &scale);
    CHECK_ERROR(err);

    size_t QxKGlobalSize[3] = {1, heightXwidth/reg_size, heightXwidth/reg_size};
    size_t QxKLocalSize[3] = {1, tile_size/reg_size, tile_size/reg_size};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->batch_matmul_scale, 3, nullptr,
                                 QxKGlobalSize, QxKLocalSize, 2, &events[2], &events[4]);
    CHECK_ERROR(err);
    /* optimized batch matmul - Q x K */

    err = clSetKernelArg(utilKernel->softmax, 0, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(utilKernel->softmax, 1, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(utilKernel->softmax, 2, sizeof(float) * WORK_GROUP_SIZE, nullptr);
    err |= clSetKernelArg(utilKernel->softmax, 3, sizeof(float) * heightXwidth, nullptr);
    err |= clSetKernelArg(utilKernel->softmax, 4, sizeof(size_t), &heightXwidth);
    CHECK_ERROR(err);

    size_t softmaxGlobalSize[1] = {heightXwidth * WORK_GROUP_SIZE};
    size_t softmaxLocalSize[1] = {WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->softmax, 1, nullptr,
                                 softmaxGlobalSize, softmaxLocalSize, 1, &events[4], &events[5]);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->permute3D_0_2_1, 0, sizeof(cl_mem), &bufferQK);
    err |= clSetKernelArg(utilKernel->permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermuteQK);
    CHECK_ERROR(err);

    size_t QKGlobalSize[3] = {1, heightXwidth, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_0_2_1, 3, nullptr,
                                 QKGlobalSize, nullptr, 1, &events[5], &events[6]);
    CHECK_ERROR(err);

    float identity = 1.f;
    /* naive batch matmul - V x QK
    err = clSetKernelArg(utilKernel->batch_matmul, 0, sizeof(cl_mem), &bufferV);
    err |= clSetKernelArg(utilKernel->batch_matmul, 1, sizeof(cl_mem), &bufferPermuteQK);
    err |= clSetKernelArg(utilKernel->batch_matmul, 2, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(utilKernel->batch_matmul, 3, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(utilKernel->batch_matmul, 4, sizeof(float), &identity);
    CHECK_ERROR(err);

    size_t VQKGlobalSize[3] = {1, in_channels, heightXwidth};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->batch_matmul, 3, nullptr,
                                 VQKGlobalSize, nullptr, 2, &events[6], &events[8]);
    CHECK_ERROR(err);
    naive batch matmul - V x QK */

    /* optimized batch matmul - V x QK*/
    if (heightXwidth % tile_size != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] heightXwidth(%ld) %% tile_size(%ld) != 0\n", __FILE__,
                            __LINE__, heightXwidth, tile_size);
        return CL_INVALID_VALUE;
    }
    if (in_channels % tile_size != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] in_channels(%ld) %% tile_size(%ld) != 0\n", __FILE__,
                            __LINE__, in_channels, tile_size);
        return CL_INVALID_VALUE;
    }
    if (heightXwidth % tile_size_k != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] heightXwidth(%ld) %% tile_size_k(%ld) != 0\n", __FILE__,
                            __LINE__, heightXwidth, tile_size_k);
        return CL_INVALID_VALUE;
    }
    err = clSetKernelArg(utilKernel->batch_matmul_scale, 0, sizeof(cl_mem), &bufferV);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 1, sizeof(cl_mem), &bufferPermuteQK);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 2, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 3, sizeof(size_t), &in_channels);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 4, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 5, sizeof(size_t), &heightXwidth);
    err |= clSetKernelArg(utilKernel->batch_matmul_scale, 6, sizeof(float), &identity);
    CHECK_ERROR(err);

    size_t VQKGlobalSize[3] = {1, in_channels/reg_size, heightXwidth/reg_size};
    size_t VQKLocalSize[3] = {1, tile_size/reg_size, tile_size/reg_size};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->batch_matmul_scale, 3, nullptr,
                                 VQKGlobalSize, VQKLocalSize, 2, &events[6], &events[8]);
    CHECK_ERROR(err);
    /*optimized batch matmul - V x QK */

    out_conv2d->init();
    err = out_conv2d->forward(bufferQ, bufferK, 1, &events[8], &events[9]);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t elemAddGlobalSize[1] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr,
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