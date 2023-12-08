//
// Created by 구현우 on 2023/12/08.
//

#include "BasicTransformerBlock.h"

#include <android/log.h>
#include "../util.h"

#define LOG_TAG "BASIC_TRANSFORMER_BLOCK"

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

BasicTransformerBlock::BasicTransformerBlock(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager, size_t headSize, size_t headDim,
        const char *layer_norm_1_weight_name, const char *layer_norm_1_bias_name,
        const char *layer_norm_2_weight_name, const char *layer_norm_2_bias_name,
        const char *layer_norm_3_weight_name, const char *layer_norm_3_bias_name,
        const char *cross_q_linear_weight_name,
        const char *cross_k_linear_weight_name,
        const char *cross_v_linear_weight_name,
        const char *cross_out_linear_weight_name, const char *cross_out_linear_bias_name
) : cmdQueue(cmdQueue), context(context) {
    layerNorm1 = new LayerNorm(context, cmdQueue, deviceId, assetManager, layer_norm_1_weight_name,
                               layer_norm_1_bias_name);
    layerNorm2 = new LayerNorm(context, cmdQueue, deviceId, assetManager, layer_norm_2_weight_name,
                               layer_norm_2_bias_name);
    layerNorm3 = new LayerNorm(context, cmdQueue, deviceId, assetManager, layer_norm_3_weight_name,
                               layer_norm_3_bias_name);
    crossAttention1 = new CrossAttention(context, cmdQueue, deviceId, assetManager, headSize, headDim,
                                         cross_q_linear_weight_name,
                                         cross_k_linear_weight_name,
                                         cross_v_linear_weight_name,
                                         cross_out_linear_weight_name, cross_out_linear_bias_name);
}

BasicTransformerBlock::~BasicTransformerBlock() {
    delete layerNorm1;
    delete layerNorm2;
    delete layerNorm3;
    delete crossAttention1;
}

cl_int BasicTransformerBlock::forward(cl_mem input, cl_mem condition, cl_mem output,
                                      cl_uint num_events_in_list, const cl_event *event_wait_list,
                                      cl_event *event) {
    cl_int err;
    cl_event event0, event1;
    cl_mem bufferNorm;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    bufferNorm = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBytes, nullptr, &err);
    CHECK_ERROR(err);

    err = layerNorm1->forward(input, bufferNorm, num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR(err);

    // max diff: 0.00000542029738426208
    // util::testBuffer(cmdQueue, bufferNorm, "unet/input_block/test/test_basic_norm1.npy");

    err = crossAttention1->forward(bufferNorm, nullptr, bufferNorm,
                                   1, &event0, &event1);
    CHECK_ERROR(err);

    clReleaseEvent(event0);
    clReleaseMemObject(bufferNorm);

    return CL_SUCCESS;
}