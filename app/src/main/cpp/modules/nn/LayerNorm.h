//
// Created by 구현우 on 2023/11/27.
//

#ifndef MY_OPENCL_LAYERNORM_H
#define MY_OPENCL_LAYERNORM_H

#include <android/asset_manager_jni.h>

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <vector>
#include "../cnpy.h"

class LayerNorm {
public:
    LayerNorm(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
              AAssetManager *assetManager, const char *weight_name, const char *bias_name);

    ~LayerNorm();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_mem bufferWeight;
    cl_mem bufferBias;
    size_t weightSize;
    size_t biasSize;
    static std::shared_ptr<_cl_program> program;
    cl_command_queue cmdQueue;
    cl_context context;
    AAssetManager *assetManager;
};


#endif //MY_OPENCL_LAYERNORM_H
