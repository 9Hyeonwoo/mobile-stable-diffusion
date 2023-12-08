//
// Created by 구현우 on 2023/12/08.
//

#include "SpatialTransformer.h"

#include <android/log.h>
#include "../util.h"

#define LOG_TAG "SPATIAL_TRANSFORMER"

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


SpatialTransformer::SpatialTransformer(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t channels,
        const char *group_norm_weight_name, const char *group_norm_bias_name,
        const char *in_linear_weight_name, const char *in_linear_bias_name
) : context(context), cmdQueue(cmdQueue), channels(channels) {
    cl_int err;
    groupNorm = new GroupNorm(context, cmdQueue, deviceId, assetManager, 32, channels, 1e-6,
                              group_norm_weight_name, group_norm_bias_name);

    projInLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                              in_linear_weight_name, in_linear_bias_name);
    transformer = new BasicTransformerBlock(context, cmdQueue, deviceId, assetManager,
                                            "unet/input_block/1/input_block_1_basic_layer_norm_1_weight.npy",
                                            "unet/input_block/1/input_block_1_basic_layer_norm_1_bias.npy",
                                            "unet/input_block/1/input_block_1_basic_layer_norm_2_weight.npy",
                                            "unet/input_block/1/input_block_1_basic_layer_norm_2_bias.npy",
                                            "unet/input_block/1/input_block_1_basic_layer_norm_3_weight.npy",
                                            "unet/input_block/1/input_block_1_basic_layer_norm_3_bias.npy",
                                            "unet/input_block/1/input_block_1_cross_q_linear_weight.npy",
                                            "unet/input_block/1/input_block_1_cross_k_linear_weight.npy",
                                            "unet/input_block/1/input_block_1_cross_v_linear_weight.npy");

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_permute_0_2_1 = clCreateKernel(program, "permute3D__0_2_1", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

SpatialTransformer::~SpatialTransformer() {
    delete groupNorm;
    delete projInLinear;
    delete transformer;
    clReleaseKernel(kernel_permute_0_2_1);
}

cl_int SpatialTransformer::forward(cl_mem input, cl_mem condition, cl_mem output,
                                   cl_uint num_events_in_list, const cl_event *event_wait_list,
                                   cl_event *event) {
    cl_int err;
    cl_event event0, event1, event2, event3;
    cl_mem bufferGroupNorm, bufferPermute;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);
    size_t inputSize = inputBytes / sizeof(float);

    bufferGroupNorm = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     inputBytes,
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferPermute = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   inputBytes,
                                   nullptr, &err);
    CHECK_ERROR(err);

    err = groupNorm->forward(input, bufferGroupNorm, num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR(err);

    // max diff: 0.00000278651714324951
    // util::testBuffer(cmdQueue, bufferGroupNorm, "unet/input_block/test/test_spatial_norm.npy");

    err = clSetKernelArg(kernel_permute_0_2_1, 0, sizeof(cl_mem), &bufferGroupNorm);
    err |= clSetKernelArg(kernel_permute_0_2_1, 1, sizeof(cl_mem), &bufferPermute);
    CHECK_ERROR(err);

    size_t permuteGlobalSize[3] = {1, channels, inputSize / channels};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute_0_2_1, 3, nullptr,
                                 permuteGlobalSize, nullptr, 1, &event0, &event1);
    CHECK_ERROR(err);

    err = projInLinear->forward(bufferPermute, bufferGroupNorm, 1, &event1, &event2);
    CHECK_ERROR(err);

    // max diff: 0.00000250339508056641
    // util::testBuffer(cmdQueue, bufferGroupNorm, "unet/input_block/test/test_spatial_proj_in.npy");

    err = transformer->forward(bufferGroupNorm, condition, bufferPermute, 1, &event2, &event3);
    CHECK_ERROR(err);

    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseMemObject(bufferGroupNorm);
    clReleaseMemObject(bufferPermute);

    return CL_SUCCESS;
}