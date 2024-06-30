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
 * Version 1: Apply register n + vector
 * Version 2: only add register n
 * Version 3: Version 2 + register c
 * Version 4: Version 3 - weight register (201ms). reg_n 은 1이 빠름.
 * Version 5: Version 4 - register n + transpose input_win + register m (102ms)
 */
#define CONV_2D_KERNEL_VERSION 5

/**
 * UNet Load Mode
 * Version 0: Initial version
 * Version 1: Load before Execute
 */
#define UNET_LOAD_MODE 1

#endif //MY_OPENCL_SETTING_H
