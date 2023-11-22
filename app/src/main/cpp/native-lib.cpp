#include <jni.h>
#include <string>
#include <android/log.h>
#include "modules/tokenizer.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) {\
      __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "BAD %d", err);\
    }

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myopencl_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jobject _assetManager) {
    std::string hello = "Hello from C++";

    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, &device, nullptr);
    CHECK_ERROR(err);

    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    SimpleTokenizer tokenizer(assetManager);
    auto a = tokenizer.tokenize("<start_of_text> wawd wdad w,. wwd qwd <end_of_text>");
    for (auto row : a) {
        for (auto i : row) {
            __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "encode: %d", i);
        }
    }

    return env->NewStringUTF(hello.c_str());
}