//
// Created by 구현우 on 2024/06/20.
//

#ifndef MY_OPENCL_UPSAMPLEKERNEL_H
#define MY_OPENCL_UPSAMPLEKERNEL_H

#include <CL/opencl.h>
#include <android/asset_manager.h>

class UpSampleKernel {
public:
    UpSampleKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager);

    ~UpSampleKernel();

    cl_kernel up_sample_nearest;
};


#endif //MY_OPENCL_UPSAMPLEKERNEL_H