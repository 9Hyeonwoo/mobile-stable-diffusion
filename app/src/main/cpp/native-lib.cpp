#include <jni.h>
#include <string>
#include <android/log.h>
#include "modules/tokenizer.h"
#include "modules/TextEncoder.h"
#include "modules/DDIMSampler.h"
#include "modules/UNetModel.h"
#include "modules/util.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#define LOG_TAG "MAIN"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

cl_context context;
cl_command_queue cmdQueue;

SimpleTokenizer *tokenizer;
TextEncoder* encoder;
DDIMSampler *sampler;

extern "C" JNIEXPORT void JNICALL
Java_com_example_myopencl_MainActivity_initOpenCL(
        JNIEnv* env,
        jobject thiz,
        jobject _assetManager) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_int err;

    err = clGetPlatformIDs(1, &platformId, nullptr);
    err |= clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
    CHECK_ERROR(err);

    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    CHECK_ERROR(err);

    cmdQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, &err);
    CHECK_ERROR(err);

    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    tokenizer = new SimpleTokenizer();
    encoder = new TextEncoder(assetManager, context, cmdQueue, deviceId);

    // auto clazz = env->FindClass("com/example/myopencl/MainActivity");
    // auto methodId = env->GetMethodID(clazz, "unet", "([FJ[F)[F");

    sampler = new DDIMSampler([](std::vector<float> x, int t, std::vector<float> c){
        // auto _result = (jfloatArray)(env->CallObjectMethod(thiz, methodId, x.data(), t, c.data()));
        // float* floatArray = env->GetFloatArrayElements(_result, nullptr);
        return x;
    });
    // int shape[3] = {4, 64, 64};
    // std::vector<float> tmp_c(77*1024, 0.f);
    // auto x = util::load_npy_file("sampler/test/test_img.npy");
    // auto x_vec = x.as_vec<float>();
    // sampler->sample(&x_vec, 50, shape, tmp_c);

    auto unet = UNetModel();

    size_t maxWorkGroupSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_GROUP_SIZE: %ld", maxWorkGroupSize);

    cl_uint maxDims;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxDims, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d", maxDims);

    size_t maxWorkItemSizes[maxDims];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDims, &maxWorkItemSizes, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_ITEM_SIZES: %ld, %ld, %ld", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

    char openclVersion[100];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, sizeof(openclVersion), &openclVersion, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_OPENCL_C_VERSION: %s", openclVersion);

    cl_ulong localMemSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_LOCAL_MEM_SIZE(32KB): %ld", localMemSize);

    cl_ulong globalMemSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_GLOBAL_MEM_SIZE(5.6GB): %ld", globalMemSize);

    cl_device_fp_config fpConfig;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fpConfig, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_SINGLE_FP_CONFIG: %ld", fpConfig);
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

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myopencl_MainActivity_encode(JNIEnv *env, jobject thiz, jlongArray _token) {
    long* longArray = env->GetLongArrayElements(_token, nullptr);
    auto token = std::vector<long>(longArray, longArray + env->GetArrayLength(_token));

    auto encodedToken = encoder->encode(token);

    jfloatArray result = env->NewFloatArray(static_cast<int>(encodedToken.size()));
    env->SetFloatArrayRegion(result, 0, static_cast<int>(encodedToken.size()), encodedToken.data());
    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myopencl_MainActivity_sample(JNIEnv *env, jobject thiz, jfloatArray _condition) {
    float* floatArray = env->GetFloatArrayElements(_condition, nullptr);
    auto condition = std::vector<float>(floatArray, floatArray + env->GetArrayLength(_condition));

    auto x = util::load_npy_file("sampler/test/test_seed_45_img.npy");
    auto x_vec = x.as_vec<float>();
    int shape[3] = {4, 64, 64};
    auto result = sampler->sample(&x_vec, 50, shape, condition);

    jfloatArray resultArray = env->NewFloatArray(static_cast<int>(result.size()));
    env->SetFloatArrayRegion(resultArray, 0, static_cast<int>(result.size()), result.data());
    return resultArray;

}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myopencl_MainActivity_destroyOpenCL(JNIEnv *env, jobject thiz) {
    delete tokenizer;
    delete encoder;
    delete sampler;

    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
}