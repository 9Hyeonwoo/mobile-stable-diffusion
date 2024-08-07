//
// Created by 구현우 on 2024/06/20.
//

#ifndef MY_OPENCL_CONVKERNEL_H
#define MY_OPENCL_CONVKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class ConvKernel {
public:
    ConvKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~ConvKernel();

    cl_kernel conv2d;
    cl_kernel im2col;
    cl_kernel conv2d_matmul;
    cl_kernel im2win;
    cl_kernel im2win_matmul;
    cl_kernel im2win_batch_matmul;
    cl_kernel im2win_reg_n_matmul;
    cl_kernel im2win_v2_matmul;
    cl_kernel im2win_channel_reg_matmul;
    cl_kernel im2win_channel_reg_v4_matmul;
    cl_kernel im2win_channel_reg_transpose_v5_matmul;
    cl_kernel im2win_transpose;
    cl_kernel im2win_channel_reg_transpose_vector_v6_matmul;
    cl_kernel im2win_channel_reg_transpose_weight_vector_v7_matmul;
    cl_kernel im2win_transpose_reorder;
    cl_kernel im2win_channel_reg_transpose_reorder_vector_v8_matmul;
};


#endif //MY_OPENCL_CONVKERNEL_H