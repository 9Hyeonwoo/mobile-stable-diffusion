//
// Created by 구현우 on 2023/12/07.
//

#include "ResBlock.h"

#include "../util.h"
#include <android/log.h>

#define LOG_TAG "RES_BLOCK"

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

ResBlock::ResBlock(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        const char *in_group_norm_weight_name, const char *in_group_norm_bias_name,
        const char *in_conv2d_weight_name, const char *in_conv2d_bias_name
) : context(context), cmdQueue(cmdQueue), assetManager(assetManager) {
    cl_int err;
    in_group_norm = new GroupNorm(context, cmdQueue, deviceId, assetManager, 32, 320,
                                  in_group_norm_weight_name, in_group_norm_bias_name);
    in_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager, in_conv2d_weight_name,
                           in_conv2d_bias_name, 1, 1);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

ResBlock::~ResBlock() {
    delete in_group_norm;
    delete in_conv2d;
    clReleaseKernel(kernel_silu);
}

cl_int ResBlock::forward(
        cl_mem input, cl_mem output,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0, event1, event2;
    cl_mem bufferInGroupNorm, bufferInConv2d;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    /* in_layers */
    bufferInGroupNorm = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       inputBytes,
                                       nullptr, &err);
    CHECK_ERROR(err);

    bufferInConv2d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    inputBytes,
                                    nullptr, &err);
    CHECK_ERROR(err);

    err = in_group_norm->forward(input, bufferInGroupNorm, num_events_in_list, event_wait_list,
                                 &event0);
    CHECK_ERROR(err);

    // max diff: 0.00000095367431640625
    // util::testBuffer(cmdQueue, bufferInputBlock_0, "unet/input_block/test/test_resblock_group_norm.npy");

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferInGroupNorm);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferInGroupNorm);
    CHECK_ERROR(err);

    size_t inSILUGlobalSize[3] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr, inSILUGlobalSize, nullptr, 1,
                                 &event0, &event1);

    err = in_conv2d->forward(bufferInGroupNorm, bufferInConv2d, 1, &event1, &event2);
    CHECK_ERROR(err);
    /* in_layers */

    // max diff: 0.00000810623168945312
    // util::testBuffer(cmdQueue, bufferInConv2d, "unet/input_block/test/test_resblock_in_layers.npy");

    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseMemObject(bufferInGroupNorm);
    clReleaseMemObject(bufferInConv2d);

    return CL_SUCCESS;
}