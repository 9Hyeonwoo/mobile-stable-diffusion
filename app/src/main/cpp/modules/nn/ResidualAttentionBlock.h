//
// Created by 구현우 on 2023/12/02.
//

#ifndef MY_OPENCL_RESIDUALATTENTIONBLOCK_H
#define MY_OPENCL_RESIDUALATTENTIONBLOCK_H

#define CL_TARGET_OPENCL_VERSION 200

#include "CL/opencl.h"
#include "LayerNorm.h"
#include "Linear.h"
#include <android/asset_manager_jni.h>

class ResidualAttentionBlock {
public:
    ResidualAttentionBlock(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                           AAssetManager *assetManager, size_t numHeads);

    ~ResidualAttentionBlock();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    LayerNorm *layerNorm0;
    Linear *attnInProj0;
    Linear *attnOutProj0;

    cl_context context;
    cl_command_queue cmdQueue;
    AAssetManager *assetManager;
    size_t numHeads;

    cl_mem bufferAttentionMask;
    cl_kernel kernel_permute3D_1_0_2;
    cl_kernel kernel_add_matmul_attention;
    cl_kernel kernel_softmax;
    cl_kernel kernel_matmul_attention;
};


#endif //MY_OPENCL_RESIDUALATTENTIONBLOCK_H
