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
        cl_context context, cl_command_queue cmdQueue,
        size_t channels, size_t context_dim, size_t headSize, size_t headDim,
        const std::string &group_norm_weight_name, const std::string &group_norm_bias_name,
        const std::string &in_linear_weight_name, const std::string &in_linear_bias_name,
        const std::string &layer_norm_1_weight_name, const std::string &layer_norm_1_bias_name,
        const std::string &layer_norm_2_weight_name, const std::string &layer_norm_2_bias_name,
        const std::string &layer_norm_3_weight_name, const std::string &layer_norm_3_bias_name,
        const std::string &cross_1_q_linear_weight_name,
        const std::string &cross_1_k_linear_weight_name,
        const std::string &cross_1_v_linear_weight_name,
        const std::string &cross_1_out_linear_weight_name, const std::string &cross_1_out_linear_bias_name,
        const std::string &cross_2_q_linear_weight_name,
        const std::string &cross_2_k_linear_weight_name,
        const std::string &cross_2_v_linear_weight_name,
        const std::string &cross_2_out_linear_weight_name, const std::string &cross_2_out_linear_bias_name,
        const std::string &ff_geglu_linear_weight_name, const std::string &ff_geglu_linear_bias_name,
        const std::string &ff_net_linear_weight_name, const std::string &ff_net_linear_bias_name,
        const std::string &out_linear_weight_name, const std::string &out_linear_bias_name,
        std::shared_ptr<LayerNormKernel> layerNormKernel,
        std::shared_ptr<LinearKernel> linearKernel,
        std::shared_ptr<UtilKernel> utilKernel,
        std::shared_ptr<CrossAttentionKernel> crossAttentionKernel,
        std::shared_ptr<GEGLUKernel> gegluKernel,
        std::shared_ptr<GroupNormKernel> groupNormKernel
) : context(context), cmdQueue(cmdQueue), channels(channels), utilKernel(utilKernel) {
    cl_int err;
    size_t inner_dim = headSize * headDim;
    groupNorm = new GroupNorm(context, cmdQueue, 32, channels, 1e-6,
                              group_norm_weight_name, group_norm_bias_name,
                              groupNormKernel);

    projInLinear = new Linear(context, cmdQueue,
                              channels, inner_dim,
                              in_linear_weight_name, in_linear_bias_name,
                              linearKernel, utilKernel);

    transformer = new BasicTransformerBlock(context, cmdQueue,
                                            inner_dim, context_dim, headSize, headDim,
                                            layer_norm_1_weight_name, layer_norm_1_bias_name,
                                            layer_norm_2_weight_name, layer_norm_2_bias_name,
                                            layer_norm_3_weight_name, layer_norm_3_bias_name,
                                            cross_1_q_linear_weight_name,
                                            cross_1_k_linear_weight_name,
                                            cross_1_v_linear_weight_name,
                                            cross_1_out_linear_weight_name,
                                            cross_1_out_linear_bias_name,
                                            cross_2_q_linear_weight_name,
                                            cross_2_k_linear_weight_name,
                                            cross_2_v_linear_weight_name,
                                            cross_2_out_linear_weight_name,
                                            cross_2_out_linear_bias_name,
                                            ff_geglu_linear_weight_name, ff_geglu_linear_bias_name,
                                            ff_net_linear_weight_name, ff_net_linear_bias_name,
                                            layerNormKernel,
                                            linearKernel,
                                            utilKernel,
                                            crossAttentionKernel,
                                            gegluKernel);

    projOutLinear = new Linear(context, cmdQueue,
                               channels, inner_dim,
                               out_linear_weight_name, out_linear_bias_name,
                               linearKernel, utilKernel);
}

SpatialTransformer::~SpatialTransformer() {
    delete groupNorm;
    delete projInLinear;
    delete transformer;
    delete projOutLinear;
}

void SpatialTransformer::init() {
    groupNorm->init();
    projInLinear->init();
    transformer->init();
    projOutLinear->init();
}

cl_int SpatialTransformer::forward(cl_mem input, cl_mem condition, cl_mem output,
                                   cl_uint num_events_in_list, const cl_event *event_wait_list,
                                   cl_event *event) {
    cl_int err;
    cl_event event0, event1, event2, event3, event4, event5;
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

    err = clSetKernelArg(utilKernel->permute3D_0_2_1, 0, sizeof(cl_mem), &bufferGroupNorm);
    err |= clSetKernelArg(utilKernel->permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermute);
    CHECK_ERROR(err);

    size_t permuteGlobalSize[3] = {1, channels, inputSize / channels};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_0_2_1, 3, nullptr,
                                 permuteGlobalSize, nullptr, 1, &event0, &event1);
    CHECK_ERROR(err);

    err = projInLinear->forward(bufferPermute, bufferGroupNorm, 1, &event1, &event2);
    CHECK_ERROR(err);

    // max diff: 0.00000250339508056641
    // util::testBuffer(cmdQueue, bufferGroupNorm, "unet/input_block/test/test_spatial_proj_in.npy");

    err = transformer->forward(bufferGroupNorm, condition, bufferPermute, 1, &event2, &event3);
    CHECK_ERROR(err);

    err = projOutLinear->forward(bufferPermute, bufferGroupNorm, 1, &event3, &event4);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->permute3D_0_2_1, 0, sizeof(cl_mem), &bufferGroupNorm);
    err |= clSetKernelArg(utilKernel->permute3D_0_2_1, 1, sizeof(cl_mem), &bufferPermute);
    CHECK_ERROR(err);

    size_t permuteGlobalSize2[3] = {1, inputSize / channels, channels};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_0_2_1, 3, nullptr,
                                 permuteGlobalSize2, nullptr, 1, &event4, &event5);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferPermute);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    size_t addGlobalSize[1] = {inputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr, addGlobalSize, nullptr,
                                 1, &event5, event);
    CHECK_ERROR(err);

    // max diff: 0.00001049041748046875
    // util::testBuffer(cmdQueue, output, "unet/input_block/test/test_spatial.npy");

    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseMemObject(bufferGroupNorm);
    clReleaseMemObject(bufferPermute);

    return CL_SUCCESS;
}