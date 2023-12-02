//
// Created by 구현우 on 2023/12/02.
//

#include "ResidualAttentionBlock.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "RESIDUAL_ATTENTION_BLOCK"
#define CONTEXT_LENGTH 77
#define EMBEDDING_SIZE 1024

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

ResidualAttentionBlock::ResidualAttentionBlock(
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t numHeads
) : context(context),
    cmdQueue(cmdQueue),
    assetManager(assetManager) {
    layerNorm0 = new LayerNorm(context, cmdQueue, deviceId, assetManager,
                               "encoder/resblock_0_layer_norm_weight_fp32.npy",
                               "encoder/resblock_0_layer_norm_bias_fp32.npy");

    attn = new MultiHeadAttention(context, cmdQueue, deviceId, assetManager, numHeads);
}

ResidualAttentionBlock::~ResidualAttentionBlock() {
    delete layerNorm0;
    delete attn;
}

cl_int ResidualAttentionBlock::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    size_t inputBytes;
    cl_event event1;
    cl_mem bufferEmbedding;

    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    bufferEmbedding = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     inputBytes,
                                     nullptr, &err);
    CHECK_ERROR(err);

    err = layerNorm0->forward(input, bufferEmbedding, num_events_in_list, event_wait_list, &event1);
    CHECK_ERROR(err);

    err = attn->forward(bufferEmbedding, bufferEmbedding, 1, &event1, event);
    CHECK_ERROR(err);

    clReleaseEvent(event1);
    clReleaseMemObject(bufferEmbedding);

    return CL_SUCCESS;
}