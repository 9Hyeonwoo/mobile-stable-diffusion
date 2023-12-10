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
        size_t channels, size_t out_channels,
        const char *in_group_norm_weight_name, const char *in_group_norm_bias_name,
        const char *in_conv2d_weight_name, const char *in_conv2d_bias_name,
        const char *embed_linear_weight_name, const char *embed_linear_bias_name,
        const char *out_group_norm_weight_name, const char *out_group_norm_bias_name,
        const char *out_conv2d_weight_name, const char *out_conv2d_bias_name
) : context(context), cmdQueue(cmdQueue) {
    cl_int err;
    in_group_norm = new GroupNorm(context, cmdQueue, deviceId, assetManager, 32, channels, 1e-5,
                                  in_group_norm_weight_name, in_group_norm_bias_name);
    in_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager, in_conv2d_weight_name,
                           in_conv2d_bias_name, 1, 1);
    embed_linear = new Linear(context, cmdQueue, deviceId, assetManager,
                                embed_linear_weight_name, embed_linear_bias_name);
    out_group_norm = new GroupNorm(context, cmdQueue, deviceId, assetManager, 32, out_channels, 1e-5,
                                   out_group_norm_weight_name, out_group_norm_bias_name);
    out_conv2d = new Conv2D(context, cmdQueue, deviceId, assetManager, out_conv2d_weight_name,
                            out_conv2d_bias_name, 1, 1);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR_THROW(err);
    kernel_chunk_add = clCreateKernel(program, "chunkwise_add", &err);
    CHECK_ERROR_THROW(err);
    kernel_elem_add = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

ResBlock::~ResBlock() {
    delete in_group_norm;
    delete in_conv2d;
    delete embed_linear;
    delete out_group_norm;
    delete out_conv2d;
    clReleaseKernel(kernel_silu);
    clReleaseKernel(kernel_chunk_add);
    clReleaseKernel(kernel_elem_add);
}

cl_int ResBlock::forward(
        cl_mem input, cl_mem embed, cl_mem output,
        cl_uint num_events_embed, const cl_event *event_wait_list_embed,
        cl_uint num_events_in_list, const cl_event *event_wait_list, cl_event *event
) {
    cl_int err;
    cl_event event0_0, event0_1, event0_2;
    cl_event event1_0;
    cl_event event2_0;
    cl_event event3_0, event3_1, event3_2;
    cl_mem bufferInGroupNorm, bufferInConv2d, bufferEmbedTemp, bufferEmbed;

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
                                 &event0_0);
    CHECK_ERROR(err);

    // max diff: 0.00000095367431640625
    // util::testBuffer(cmdQueue, bufferInputBlock_0, "unet/input_block/test/test_resblock_group_norm.npy");

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferInGroupNorm);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferInGroupNorm);
    CHECK_ERROR(err);

    size_t inSILUGlobalSize[3] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr, inSILUGlobalSize, nullptr, 1,
                                 &event0_0, &event0_1);

    err = in_conv2d->forward(bufferInGroupNorm, bufferInConv2d, 1, &event0_1, &event0_2);
    CHECK_ERROR(err);

    // max diff: 0.00000810623168945312
    // util::testBuffer(cmdQueue, bufferInConv2d, "unet/input_block/test/test_resblock_in_layers.npy");
    /* in_layers */

    /* emb_layers */
    size_t embedBytes;
    err = clGetMemObjectInfo(embed, CL_MEM_SIZE, sizeof(size_t), &embedBytes, nullptr);
    CHECK_ERROR(err);

    bufferEmbedTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     embedBytes,
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferEmbed = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * 320,
                                 nullptr, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &embed);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferEmbedTemp);
    CHECK_ERROR(err);

    size_t embSILUGlobalSize[1] = {embedBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr, embSILUGlobalSize, nullptr,
                                 num_events_embed, event_wait_list_embed, &event1_0);
    CHECK_ERROR(err);

    err = embed_linear->forward(bufferEmbedTemp, bufferEmbed, 1, &event1_0, &event0_2);
    CHECK_ERROR(err);

    // max diff: 0.00001716613769531250
    // util::testBuffer(cmdQueue, bufferEmbed, "unet/input_block/test/test_resblock_embed.npy");
    /* emb_layers */

    size_t chunkSize = inputBytes / sizeof(float) / 320;
    err = clSetKernelArg(kernel_chunk_add, 0, sizeof(cl_mem), &bufferInConv2d);
    err |= clSetKernelArg(kernel_chunk_add, 1, sizeof(cl_mem), &bufferEmbed);
    err |= clSetKernelArg(kernel_chunk_add, 2, sizeof(cl_mem), &bufferInConv2d);
    err |= clSetKernelArg(kernel_chunk_add, 3, sizeof(size_t), &chunkSize);
    CHECK_ERROR(err);

    size_t chunkAddGlobalSize[1] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_chunk_add, 1, nullptr, chunkAddGlobalSize,
                                 nullptr,
                                 2, &event0_2, &event2_0);
    CHECK_ERROR(err);

    // max diff: 0.00002098083496093750
    // util::testBuffer(cmdQueue, bufferInConv2d, "unet/input_block/test/test_resblock_chunk_add.npy");

    /* out_layers */
    err = out_group_norm->forward(bufferInConv2d, bufferInGroupNorm, 1, &event2_0, &event3_0);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_silu, 0, sizeof(cl_mem), &bufferInGroupNorm);
    err |= clSetKernelArg(kernel_silu, 1, sizeof(cl_mem), &bufferInGroupNorm);
    CHECK_ERROR(err);

    size_t outSILUGlobalSize[3] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_silu, 1, nullptr, outSILUGlobalSize, nullptr, 1,
                                 &event3_0, &event3_1);
    CHECK_ERROR(err);

    err = out_conv2d->forward(bufferInGroupNorm, bufferInConv2d, 1, &event3_1, &event3_2);
    CHECK_ERROR(err);

    // max diff: 0.00000953674316406250
    // util::testBuffer(cmdQueue, bufferInConv2d, "unet/input_block/test/test_resblock_out_layers.npy");
    /* out_layers */

    /* skip_connection */
    err = clSetKernelArg(kernel_elem_add, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_elem_add, 1, sizeof(cl_mem), &bufferInConv2d);
    err |= clSetKernelArg(kernel_elem_add, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t elemAddGlobalSize[1] = {inputBytes / sizeof(float)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_elem_add, 1, nullptr, elemAddGlobalSize, nullptr,
                                 1, &event3_2, event);
    CHECK_ERROR(err);

    // max diff: 0.00000953674316406250
    // util::testBuffer(cmdQueue, output, "unet/input_block/test/test_resblock_skip_connection.npy");
    /* skip_connection */

    clReleaseEvent(event0_0);
    clReleaseEvent(event0_1);
    clReleaseEvent(event0_2);
    clReleaseEvent(event1_0);
    clReleaseEvent(event2_0);
    clReleaseEvent(event3_0);
    clReleaseEvent(event3_1);
    clReleaseEvent(event3_2);
    clReleaseMemObject(bufferInGroupNorm);
    clReleaseMemObject(bufferInConv2d);
    clReleaseMemObject(bufferEmbedTemp);
    clReleaseMemObject(bufferEmbed);

    return CL_SUCCESS;
}