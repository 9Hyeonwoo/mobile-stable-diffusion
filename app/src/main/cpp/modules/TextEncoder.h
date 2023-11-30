//
// Created by 구현우 on 2023/11/24.
//

#ifndef MY_OPENCL_TEXTENCODER_H
#define MY_OPENCL_TEXTENCODER_H

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "cnpy.h"
#include <vector>
#include <stdio.h>
#include "nn/LayerNorm.h"

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>

/*
 * kernel verification
 * elemwise_add : Checked! (2023/11/29)
 * permute3D__1_0_2 : Checked! (2023/11/29)
 */
class TextEncoder {
public:
    TextEncoder(AAssetManager *assetManager, cl_context context, cl_command_queue cmdQueue,
                cl_device_id deviceId);

    ~TextEncoder();

    std::vector<float> encode(const std::vector<long> &token);

private:
    std::vector<float> token_embedding(const std::vector<long> &token);

    // Checked! (2023/11/29)
    void testEmbedding(const std::vector<long> &token);

    cl_context context;
    cl_command_queue cmdQueue;
    cl_device_id deviceId;
    AAssetManager *assetManager;

    cnpy::NpyArray *embedding;

    LayerNorm *layerNorm0;

    // Checked! (2023/11/29)
    // positional_embedding.shape = (CONTEXT_LENGTH, EMBEDDING_SIZE)
    cl_mem bufferPositionalEmbedding;
};


#endif //MY_OPENCL_TEXTENCODER_H
