//
// Created by 구현우 on 2024/04/28.
//

#ifndef MY_OPENCL_SETTING_H
#define MY_OPENCL_SETTING_H

#define LINEAR_KERNEL_VERSION 4

#define CROSS_ATTENTION_KERNEL_VERSION 2

/**
 * Conv2D
 * Version 0: Initial version
 * Version 1: Apply register n kernel
 */
#define CONV_2D_KERNEL_VERSION 1

/**
 * UNet Load Mode
 * Version 0: Initial version
 * Version 1: Load before Execute
 */
#define UNET_LOAD_MODE 1

#endif //MY_OPENCL_SETTING_H
