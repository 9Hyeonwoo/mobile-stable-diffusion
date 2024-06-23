//
// Created by 구현우 on 2024/06/20.
//

#ifndef MY_OPENCL_CROSSATTENTIONKERNEL_H
#define MY_OPENCL_CROSSATTENTIONKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class CrossAttentionKernel {
public:
    CrossAttentionKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~CrossAttentionKernel();

    cl_kernel einsum_bik_bjk_bij;
    cl_kernel einsum_bij_bjk_bik;

    cl_kernel optimized_einsum_bik_bjk_bij;
    cl_kernel optimized_einsum_bik_bkj_bij;
    cl_kernel optimized_einsum_bik_bkj_bij_general;
};


#endif //MY_OPENCL_CROSSATTENTIONKERNEL_H