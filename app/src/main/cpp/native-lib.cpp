#include <jni.h>
#include <string>
#include <android/log.h>
#include "modules/tokenizer.h"
#include "modules/TextEncoder.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) {\
      __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "BAD %d", err);\
    }

SimpleTokenizer *tokenizer;
TextEncoder* encoder;

extern "C" JNIEXPORT void JNICALL
Java_com_example_myopencl_MainActivity_initTokenizer(
        JNIEnv* env,
        jobject /* this */,
        jobject _assetManager) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, &device, nullptr);
    CHECK_ERROR(err);

    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    tokenizer = new SimpleTokenizer(assetManager);
    encoder = new TextEncoder(assetManager);
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_com_example_myopencl_MainActivity_tokenize(JNIEnv *env, jobject thiz, jstring _text) {
    if (tokenizer == nullptr) {
        __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "tokenizer is null");
        return nullptr;
    }

    const char *text = env->GetStringUTFChars(_text, nullptr);
    auto result = tokenizer->tokenize(text);
    for (const auto i : result) {
        __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "encode: %ld", i);
    }
    env->ReleaseStringUTFChars(_text, text);

    jlongArray resultArray = env->NewLongArray(static_cast<int>(result.size()));
    env->SetLongArrayRegion(resultArray, 0, static_cast<int>(result.size()), result.data());

    return resultArray;
}