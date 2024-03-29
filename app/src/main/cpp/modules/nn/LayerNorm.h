//
// Created by 구현우 on 2023/11/27.
//

#ifndef MY_OPENCL_LAYERNORM_H
#define MY_OPENCL_LAYERNORM_H

#include <android/asset_manager_jni.h>

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>
#include <string>

class LayerNorm {
public:
    LayerNorm(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
              AAssetManager *assetManager, size_t dim,
              const std::string &weight_name, const std::string &bias_name);

    ~LayerNorm();

    void init();

    cl_int forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                   const cl_event *event_wait_list, cl_event *event);

private:
    cl_mem bufferWeight;
    cl_mem bufferBias;
    size_t weightSize;
    size_t biasSize;
    cl_command_queue cmdQueue;
    cl_context context;

    cl_kernel kernel_mean;
    cl_kernel kernel_var;
    cl_kernel kernel_norm;

    cl_event event_init_weight;
    cl_event event_init_bias;

    const std::string weight_name;
    const std::string bias_name;
};


#endif //MY_OPENCL_LAYERNORM_H
