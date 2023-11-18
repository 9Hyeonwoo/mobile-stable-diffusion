#include <jni.h>
#include <string>
#include <android/log.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) {\
      __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "BAD %d", err);\
    }

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myopencl_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERROR(err);

    return env->NewStringUTF(hello.c_str());
}