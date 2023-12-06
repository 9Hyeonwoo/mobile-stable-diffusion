//
// Created by 구현우 on 2023/11/25.
//

#ifndef MY_OPENCL_UTIL_H
#define MY_OPENCL_UTIL_H

#include <vector>
#include <android/asset_manager_jni.h>

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>

#include "cnpy.h"

namespace util {
    std::vector<float> *
    permute3D(const std::vector<float> &vec, const int shape[3], const int dimensions[3]);

    cl_program create_and_build_program_with_source(cl_context context, cl_device_id device,
                                                    AAssetManager *assetManager,
                                                    const char *file_name);

    cnpy::NpyArray load_npy_file(AAssetManager *assetManager, const char *filename);

    cnpy::NpyArray load_npy_file(const char *filename);

    void testBuffer(AAssetManager *assetManager, cl_command_queue cmdQueue, cl_mem buffer,
                    const char *filename);

    void testBuffer(cl_command_queue cmdQueue, cl_mem buffer, const char *filename);

    void testBuffer(std::vector<float> result, const char *filename);
}
#endif //MY_OPENCL_UTIL_H
