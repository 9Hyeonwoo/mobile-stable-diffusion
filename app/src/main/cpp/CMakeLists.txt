# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("myopencl")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp
        modules/tokenizer.cpp
        modules/TextEncoder.cpp
        modules/cnpy.cpp
        modules/util.cpp
        modules/nn/LayerNorm.cpp
        modules/nn/Linear.cpp
        modules/nn/ResidualAttentionBlock.cpp
        modules/nn/MultiHeadAttention.cpp
        modules/DDIMSampler.cpp
        modules/UNetModel.cpp
        modules/nn/Conv2D.cpp
        modules/nn/GroupNorm.cpp
        modules/nn/ResBlock.cpp
        modules/nn/SpatialTransformer.cpp
        modules/nn/BasicTransformerBlock.cpp
        modules/nn/CrossAttention.cpp
        modules/nn/FeedForward.cpp
        modules/nn/GEGLU.cpp
        modules/nn/UpSample.cpp
        modules/Decoder.cpp
        modules/nn/AttnBlock.cpp
        modules/kernel/unit/LayerNormKernel.cpp
        modules/kernel/unit/LinearKernel.cpp
        modules/kernel/unit/MultiHeadAttentionKernel.cpp
        modules/kernel/unit/UtilKernel.cpp
        modules/kernel/unit/ConvKernel.cpp
        modules/kernel/unit/CrossAttentionKernel.cpp
        modules/kernel/unit/GEGLUKernel.cpp
)

# add libraries for OpenCL
include_directories( ${CMAKE_SOURCE_DIR}/include )

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_library( OpenCL
        SHARED
        IMPORTED )

set_target_properties(
        OpenCL
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libGLES_mali.so )

add_library( cpp
        SHARED
        IMPORTED )

set_target_properties(
        cpp
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libc++.so )

add_library( utils
        SHARED
        IMPORTED )

set_target_properties(
        utils
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libutils.so )

add_library( gra
        SHARED
        IMPORTED )

set_target_properties(
        gra
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libgralloctypes.so )

add_library( ion
        SHARED
        IMPORTED )

set_target_properties(
        ion
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libion_exynos.so )

add_library( base
        SHARED
        IMPORTED )

set_target_properties(
        base
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libbase.so )

add_library( cutils
        SHARED
        IMPORTED )

set_target_properties(
        cutils
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libcutils.so )

add_library( mapper
        SHARED
        IMPORTED )

set_target_properties(
        mapper
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.graphics.mapper@4.0.so )

add_library( hidl
        SHARED
        IMPORTED )

set_target_properties(
        hidl
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libhidlbase.so )

add_library( hardware
        SHARED
        IMPORTED )

set_target_properties(
        hardware
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libhardware.so )

add_library( vndksupport
        SHARED
        IMPORTED )

set_target_properties(
        vndksupport
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libvndksupport.so )

add_library( commonNdk
        SHARED
        IMPORTED )

set_target_properties(
        commonNdk
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.graphics.common-V3-ndk.so )

add_library( common
        SHARED
        IMPORTED )

set_target_properties(
        common
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.graphics.common@1.0.so )

add_library( common1
        SHARED
        IMPORTED )

set_target_properties(
        common1
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.graphics.common@1.1.so )

add_library( common2
        SHARED
        IMPORTED )

set_target_properties(
        common2
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.graphics.common@1.2.so )

add_library( common2Ndk
        SHARED
        IMPORTED )

set_target_properties(
        common2Ndk
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/android.hardware.common-V2-ndk.so )

add_library( dlAndroid
        SHARED
        IMPORTED )

set_target_properties(
        dlAndroid
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/libdl_android.so )

add_library( ldA
        SHARED
        IMPORTED )

set_target_properties(
        ldA
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}/ld-android.so )

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # OpenCL libraries
        OpenCL
        utils
        gra
        base
        ion
        cutils
        mapper
        hidl
        hardware
        cpp
        commonNdk
        common
        common1
        common2
        common2Ndk
        vndksupport
        dlAndroid
        ldA
        # List libraries link to the target library
        ${ZLIB_LIBRARIES}
        android
        log)