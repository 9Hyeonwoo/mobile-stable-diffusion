//
// Created by 구현우 on 2024/04/20.
//

#ifndef MY_OPENCL_LINEARKERNEL_H
#define MY_OPENCL_LINEARKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class LinearKernel {
public:
    LinearKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~LinearKernel();

    cl_kernel naive_linear;
    cl_kernel register_linear;
    cl_kernel tile_linear;
    cl_kernel tile_reg_n_linear;
};


#endif //MY_OPENCL_LINEARKERNEL_H
