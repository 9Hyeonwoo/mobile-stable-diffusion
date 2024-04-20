//
// Created by 구현우 on 2024/04/20.
//

#ifndef MY_OPENCL_MULTIHEADATTENTIONKERNEL_H
#define MY_OPENCL_MULTIHEADATTENTIONKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class MultiHeadAttentionKernel {
public:
    MultiHeadAttentionKernel(cl_context context, cl_device_id deviceId,
                             AAssetManager *assetManager);

    ~MultiHeadAttentionKernel();

    cl_kernel add_matmul_attention;
    cl_kernel softmax;
    cl_kernel matmul_attention;
    cl_kernel batch_matmul_mask;
    cl_kernel batch_matmul;
};


#endif //MY_OPENCL_MULTIHEADATTENTIONKERNEL_H
