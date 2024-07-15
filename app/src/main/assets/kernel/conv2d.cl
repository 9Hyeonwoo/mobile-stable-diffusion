__kernel void conv2d(__global float *input,
                     __global float *weight,
                     __global float *bias,
                     __global float *output,
                     const int input_size,
                     const int kernel_channel_size,
                     const int kernel_size,
                     const int stride,
                     const int padding)
{
    int channel = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);
    int output_size = get_global_size(1);
    int kernel_half = kernel_size / 2;

    float sum = 0.0f;
    for (int kc = 0; kc < kernel_channel_size; kc++) {
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                int input_i = i * stride + ki - padding;
                int input_j = j * stride + kj - padding;
                if (input_i < 0 || input_i >= input_size || input_j < 0 || input_j >= input_size) {
                    continue;
                }

                int input_index = kc * input_size * input_size + input_i * input_size + input_j;
                int kernel_index = (channel * kernel_channel_size * kernel_size * kernel_size) + (kc * kernel_size * kernel_size) + (ki * kernel_size) + (kj);

                sum += input[input_index] * weight[kernel_index];
            }
        }
    }

    int output_index =  channel * output_size * output_size + i * output_size + j;
    output[output_index] = sum + bias[channel];
}

__kernel void im2col(
    const int n,
    __global const float* data_im,
    const int data_im_off,
    const int height,
    const int width,
    const int kernel_size,
    const int padding,
    const int stride,
    const int height_col,
    const int width_col,
    __global float* data_col,
    const int data_col_off
) {
    // im = (4, 64, 64) kernel = (320, 4, 3, 3) padding = (1) stride = (1) => col = (4*3*3, 64*64)
    // n = channels * height_col * width_col
    // n x kernel_size x kernel_size
    for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_size * kernel_size;
        const int h_offset = h_col * stride - padding;
        const int w_offset = w_col * stride - padding;

        __global float* data_col_ptr = data_col + data_col_off;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

        __global const float* data_im_ptr = data_im + data_im_off;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        for (int i = 0; i < kernel_size; ++i) {
          for (int j = 0; j < kernel_size; ++j) {
            int h_im = h_offset + i;
            int w_im = w_offset + j;
            *data_col_ptr =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * width + j] : 0.0f;
            data_col_ptr += height_col * width_col;
          }
        }
    }
}

__kernel void conv2d_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_col,
    __global float *output,
    const int M,
    const int N,
    const int K
) {
    // weight = (320, 4, 3, 3) bias = (320), input_col = (4*3*3, 64*64) => output = (320, 64*64)
    int output_size = M * N;

    for (int index = get_global_id(0); index < output_size; index += get_global_size(0)) {
        int i = index / N;
        int j = index % N;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // weight(M, K) * input_col(K, N) = output(M, N)
            sum = weight[i * K + k] * input_col[k * N + j] + sum;
        }
        output[index] = sum + bias[i];
    }
}

__kernel void im2win(
    const int n,
    __global const float* data_im,
    const int data_im_off,
    const int height,
    const int width,
    const int kernel_size,
    const int padding,
    const int stride,
    const int height_win,
    const int width_win,
    __global float* data_win,
    const int data_win_off
) {
    // im = (4, 64, 64) kernel = (320, 4, 3, 3) padding = (1) stride = (1) => col = (64, 4*3*64)
    // n = channels * height_win * (width + 2*padding)
    // n x kernel_size
    int width_pad = width + 2 * padding;
    for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
        const int h_index = index / width_pad;
        const int h_win = h_index % height_win;
        const int w_pad = index % width_pad;
        const int w_win = w_pad * kernel_size;
        const int c_im = h_index / height_win;
        const int h_offset = h_win * stride - padding;
        const int w_offset = w_pad - padding;

        // const int width_win = (width + 2 * padding) * kernel_size;
        __global float* data_win_ptr = data_win + data_win_off;
        data_win_ptr += (c_im * height_win + h_win) * width_win + w_win;

        __global const float* data_im_ptr = data_im + data_im_off;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        for (int i = 0; i < kernel_size; ++i) {
            int h_im = h_offset + i;
            int w_im = w_offset;
            *data_win_ptr =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * width] : 0.0f;
            data_win_ptr += 1;
        }
    }
}

__kernel void im2win_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int C,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    int output_size = C * M * N;
    for (int index = get_global_id(0); index < output_size; index += get_global_size(0)) {
        const int c = index / (M * N);
        const int m = (index / N) % M;
        const int n = index % N;

        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channel; c_in++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int weight_index = (((c * in_channel + c_in) * kernel_size) + i) * kernel_size + j;
                    // width_win = (width + 2 * padding) * kernel_size
                    int input_index = ((c_in * M + m) * width_win) + ((n * stride + j) * kernel_size + i);
                    sum += weight[weight_index] * input_win[input_index];
                }
            }
        }
        output[index] = sum + bias[c];
    }
}

__kernel void im2win_batch_matmul(
    __global const float *input_win, // input_win = (in_channel, M, width_pad * kernel_size)
    __global const float *weight, // weight = (C, in_channel, kernel_size, kernel_size)
    __global const float *bias, // bias = (C)
    __global float* output,
    const int M,
    const int N,
    const int in_channel,
    const int width_win,
    const int kernel_size, // kernel_size = {1, 3}
    const int stride, // stride = {1, 2}
    __local float* input_sub,
    __local float* weight_sub,
    const int tile_size_n,
    const int tile_size_k
) {

    // tile_size(batch, m, n) = (1, 1, 128)
    // reg_size, input_reg[kernel_size^2 + (reg_size_n - 1) * stride * kernel_size]
    __constant int reg_size_m = 1; // fixed value
    __constant int reg_size_n = 4;
    __constant int reg_size_input_max = 3*3 + (reg_size_n-1)*2*3;
    const int reg_size_input = kernel_size*kernel_size + (reg_size_n-1)*stride*kernel_size;
    // const int tile_size_k = 16; // in_channel % tile_size_k == 0
    const int local_m = get_local_id(1);
    const int local_n = get_local_id(2);
    const int local_size_m = get_local_size(1);
    const int local_size_n = get_local_size(2);
    const int tile_size_m = 1;
    // const int tile_size_n = 128;
    const int input_tile_size_n = kernel_size*kernel_size + (tile_size_n -1)*stride*kernel_size;
    const int input_tile_size = tile_size_k * tile_size_m * input_tile_size_n;
    const int offset_m = get_group_id(1) * get_local_size(1) * reg_size_m ;
    const int offset_output_n = get_group_id(2) * get_local_size(2) * reg_size_n;
    const int offset_input_n = get_group_id(2) * get_local_size(2) * reg_size_n * stride * kernel_size;
    const int batch = get_global_id(0);
    const int offset_batch_weight = batch * in_channel * kernel_size * kernel_size;
    const int offset_batch_output = batch * M * N ;

    // Allocate register space
    // float weight_reg;
    // float input_reg[reg_size_input_max];

    float weight_reg[3 * 3];
    float input_reg;
    float acc[reg_size_n];

    // Initialise the accumulation registers
    for (int w=0; w<reg_size_n; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    int numTiles = in_channel/tile_size_k;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int id=local_m*local_size_n + local_n; id<input_tile_size; id+=local_size_m*local_size_n) {
            int n = id % input_tile_size_n;
            int tmp = id / input_tile_size_n;
            int m = tmp % tile_size_m;
            int k = tmp / tile_size_m;

            int sub_index = (k * tile_size_m + m) * input_tile_size_n + (n);
            int input_index = ((k + t*tile_size_k) * M * width_win) + ((offset_m + m) * width_win) + (offset_input_n + n);
            if (input_index < in_channel * M * width_win) {
                input_sub[sub_index] = input_win[input_index];
            } else {
                break;
            }
        }
        for (int id=local_m*local_size_n + local_n; id<(tile_size_k * kernel_size * kernel_size); id+=local_size_m*local_size_n) {
            int n = id % kernel_size;
            int tmp = id / kernel_size;
            int m = tmp % kernel_size;
            int k = tmp / kernel_size;

            int sub_index = (k * kernel_size + m) * kernel_size + (n);
            int weight_index = ((k + t*tile_size_k) * kernel_size * kernel_size) + (m * kernel_size) + (n);
            if (weight_index < in_channel * kernel_size * kernel_size) {
                weight_sub[sub_index] = weight[weight_index + offset_batch_weight];
            } else {
                break;
            }
        }
        // global version
        // __global const float* Asub_global = A + offset_m * K + t * tile_size_k + offset_batch_A;
        // __global const float* Bsub_global = B + (t * tile_size_k) * N + offset_n + offset_batch_B;

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        /*
        for (int k=0; k<tile_size_k; k++) {
            // Cache the values of input_reg in registers
            for (int wn=0; wn<reg_size_input; wn++) {
                int col = (local_n*reg_size_n * stride * kernel_size) + wn;
                input_reg[wn] = input_sub[k*tile_size_m*input_tile_size_n + col];
            }

            // Perform the computation
            for (int kernel_i=0; kernel_i < kernel_size; kernel_i++) {
                for (int kernel_j=0; kernel_j<kernel_size; kernel_j++) {
                    weight_reg = weight_sub[k*kernel_size*kernel_size + (kernel_i*kernel_size) + kernel_j];
                    for (int rn=0; rn<reg_size_n; rn++) {
                        acc[rn] += weight_reg * input_reg[rn*stride*kernel_size + (kernel_j*kernel_size) + kernel_i];
                    }
                }
            }
        }
        */

        for (int k=0; k<tile_size_k; k++) {
            // Cache the values of input_reg in registers
            /*
            for (int wn=0; wn < kernel_size*kernel_size; wn++) {
                // int kernel_i = wn % kernel_size;
                // int kernel_j = wn / kernel_size;
                int kernel_j = wn % kernel_size;
                int kernel_i = wn / kernel_size;
                weight_reg[wn] = weight_sub[k*kernel_size*kernel_size + (kernel_i*kernel_size) + kernel_j];
            }
            */

            /*
            for (int kernel_i = 0; kernel_i < kernel_size; kernel_i++) {
                for (int kernel_j = 0; kernel_j < kernel_size; kernel_j++) {
                    weight_reg[kernel_j * kernel_size + kernel_i] = weight_sub[k*kernel_size*kernel_size + (kernel_i*kernel_size) + kernel_j];
                }
            }
            */

            for (int wn = 0; wn < kernel_size*kernel_size; wn++) {
                weight_reg[wn] = weight_sub[k*kernel_size*kernel_size + wn];
            }

            // Perform the computation
            /*
            for (int rn=0; rn<reg_size_n; rn++) {
                for (int wn=0; wn < kernel_size*kernel_size; wn++) {
                    int kernel_j = wn % kernel_size;
                    int kernel_i = wn / kernel_size;
                    int col = (local_n + rn*local_size_n) * stride * kernel_size + kernel_j*kernel_size + kernel_i;
                    input_reg = input_sub[k*tile_size_m*input_tile_size_n + col];
                    acc[rn] += weight_reg[kernel_i * kernel_size + kernel_j] * input_reg;
                }
            }
            */

            for (int rn=0; rn<reg_size_n; rn++) {
                int col_offset = (local_n + rn*local_size_n) * stride * kernel_size;
                for (int kernel_j=0; kernel_j < kernel_size; kernel_j++) {
                    for (int kernel_i=0; kernel_i < kernel_size; kernel_i++) {
                        int col = col_offset + kernel_j*kernel_size + kernel_i;
                        input_reg = input_sub[k*tile_size_m*input_tile_size_n + col];
                        acc[rn] += weight_reg[kernel_i * kernel_size + kernel_j] * input_reg;
                    }
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int rn=0; rn<reg_size_n; rn++) {
        int global_m = offset_m + local_m;
        int global_n = offset_output_n + (local_n +  rn * local_size_n);
        output[global_m*N + global_n + offset_batch_output] = acc[rn] + bias[batch];
    }
}

#define WIDTH 4
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#elif WIDTH == 16
    typedef float16 floatX;
#endif

__kernel void im2win_reg_n_matmul(
    __global const floatX *weight,
    __global const float *bias,
    __global const floatX *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_n = 16;
    // global = { C, M * N } / local = { 1, 32 }
    const int local_size_mn = get_local_size(1);
    const int c = get_global_id(0);
    const int mn = get_group_id(1) * local_size_mn * reg_size_n + get_local_id(1);
    const int m = mn / N;
    const int n = mn % N;

    float sum[reg_size_n];
    for (int i = 0; i < reg_size_n; i++) {
        sum[i] = 0.0f;
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        float weight_reg[9];
        int weight_index = (c * in_channel + c_in) * kernel_size * kernel_size;
        floatX tmp = weight[weight_index / WIDTH];
        for (int ij = 0; ij < kernel_size * kernel_size; ij++, weight_index++) {
            if (ij != 0 && weight_index % WIDTH == 0) {
                tmp = weight[weight_index / WIDTH];
            }
            int f_i = ij / kernel_size;
            int f_j = ij % kernel_size;
#if WIDTH == 1
            weight_reg[f_j * kernel_size + f_i] = tmp;
#elif WIDTH == 2
            switch (weight_index % WIDTH) {
                case 0:
                    weight_reg[f_j * kernel_size + f_i] = tmp.x;
                    break;
                case 1:
                    weight_reg[f_j * kernel_size + f_i] = tmp.y;
                    break;
            }
#elif WIDTH == 4
            switch (weight_index % WIDTH) {
                case 0:
                    weight_reg[f_j * kernel_size + f_i] = tmp.x;
                    break;
                case 1:
                    weight_reg[f_j * kernel_size + f_i] = tmp.y;
                    break;
                case 2:
                    weight_reg[f_j * kernel_size + f_i] = tmp.z;
                    break;
                case 3:
                    weight_reg[f_j * kernel_size + f_i] = tmp.w;
                    break;
            }
#elif WIDTH == 8
            switch (weight_index % WIDTH) {
                case 0:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s0;
                    break;
                case 1:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s1;
                    break;
                case 2:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s2;
                    break;
                case 3:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s3;
                    break;
                case 4:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s4;
                    break;
                case 5:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s5;
                    break;
                case 6:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s6;
                    break;
                case 7:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s7;
                    break;
            }
#elif WIDTH == 16
            switch (weight_index % WIDTH) {
                case 0:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s0;
                    break;
                case 1:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s1;
                    break;
                case 2:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s2;
                    break;
                case 3:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s3;
                    break;
                case 4:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s4;
                    break;
                case 5:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s5;
                    break;
                case 6:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s6;
                    break;
                case 7:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s7;
                    break;
                case 8:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s8;
                    break;
                case 9:
                    weight_reg[f_j * kernel_size + f_i] = tmp.s9;
                    break;
                case 10:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sA;
                    break;
                case 11:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sB;
                    break;
                case 12:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sC;
                    break;
                case 13:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sD;
                    break;
                case 14:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sE;
                    break;
                case 15:
                    weight_reg[f_j * kernel_size + f_i] = tmp.sF;
                    break;
            }
#endif
        }

        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            const int i_mn = mn + reg_n * local_size_mn;
            const int i_m = i_mn / N;
            const int i_n = i_mn % N;
            int input_index = ((c_in * M + i_m) * width_win) + ((i_n) * stride * kernel_size);
            tmp = input_win[input_index / WIDTH];
            for (int ij = 0; ij < kernel_size * kernel_size; ij++, input_index++) {
                if (ij != 0 && input_index % WIDTH == 0) {
                    tmp = input_win[input_index / WIDTH];
                }
#if WIDTH == 1
                sum[reg_n] += weight_reg[ij] * tmp;
#elif WIDTH == 2
                switch (input_index % WIDTH) {
                    case 0:
                        sum[reg_n] += weight_reg[ij] * tmp.x;
                        break;
                    case 1:
                        sum[reg_n] += weight_reg[ij] * tmp.y;
                        break;
                }
#elif WIDTH == 4
                switch (input_index % WIDTH) {
                    case 0:
                        sum[reg_n] += weight_reg[ij] * tmp.x;
                        break;
                    case 1:
                        sum[reg_n] += weight_reg[ij] * tmp.y;
                        break;
                    case 2:
                        sum[reg_n] += weight_reg[ij] * tmp.z;
                        break;
                    case 3:
                        sum[reg_n] += weight_reg[ij] * tmp.w;
                        break;
                }
#elif WIDTH == 8
                switch (input_index % WIDTH) {
                    case 0:
                        sum[reg_n] += weight_reg[ij] * tmp.s0;
                        break;
                    case 1:
                        sum[reg_n] += weight_reg[ij] * tmp.s1;
                        break;
                    case 2:
                        sum[reg_n] += weight_reg[ij] * tmp.s2;
                        break;
                    case 3:
                        sum[reg_n] += weight_reg[ij] * tmp.s3;
                        break;
                    case 4:
                        sum[reg_n] += weight_reg[ij] * tmp.s4;
                        break;
                    case 5:
                        sum[reg_n] += weight_reg[ij] * tmp.s5;
                        break;
                    case 6:
                        sum[reg_n] += weight_reg[ij] * tmp.s6;
                        break;
                    case 7:
                        sum[reg_n] += weight_reg[ij] * tmp.s7;
                        break;
                }
#elif WIDTH == 16
                switch (input_index % WIDTH) {
                    case 0:
                        sum[reg_n] += weight_reg[ij] * tmp.s0;
                        break;
                    case 1:
                        sum[reg_n] += weight_reg[ij] * tmp.s1;
                        break;
                    case 2:
                        sum[reg_n] += weight_reg[ij] * tmp.s2;
                        break;
                    case 3:
                        sum[reg_n] += weight_reg[ij] * tmp.s3;
                        break;
                    case 4:
                        sum[reg_n] += weight_reg[ij] * tmp.s4;
                        break;
                    case 5:
                        sum[reg_n] += weight_reg[ij] * tmp.s5;
                        break;
                    case 6:
                        sum[reg_n] += weight_reg[ij] * tmp.s6;
                        break;
                    case 7:
                        sum[reg_n] += weight_reg[ij] * tmp.s7;
                        break;
                    case 8:
                        sum[reg_n] += weight_reg[ij] * tmp.s8;
                        break;
                    case 9:
                        sum[reg_n] += weight_reg[ij] * tmp.s9;
                        break;
                    case 10:
                        sum[reg_n] += weight_reg[ij] * tmp.sA;
                        break;
                    case 11:
                        sum[reg_n] += weight_reg[ij] * tmp.sB;
                        break;
                    case 12:
                        sum[reg_n] += weight_reg[ij] * tmp.sC;
                        break;
                    case 13:
                        sum[reg_n] += weight_reg[ij] * tmp.sD;
                        break;
                    case 14:
                        sum[reg_n] += weight_reg[ij] * tmp.sE;
                        break;
                    case 15:
                        sum[reg_n] += weight_reg[ij] * tmp.sF;
                        break;
                }
#endif
            }
        }
    }

    for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
        output[(c * M + m) * N + n + reg_n * local_size_mn] = sum[reg_n] + bias[c];
    }
}

__kernel void im2win_v2_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_n = 32;
    const int c = get_global_id(0);
    const int mn = get_group_id(1) * get_local_size(1) * reg_size_n + get_local_id(1);
    const int m = mn / N;
    const int n = mn % N;

    float sum[reg_size_n];
    for (int i = 0; i < reg_size_n; i++) {
        sum[i] = 0.0f;
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        float weight_reg[9];
        for (int ij = 0; ij < kernel_size * kernel_size; ij++) {
            int f_i = ij / kernel_size;
            int f_j = ij % kernel_size;
            weight_reg[f_j * kernel_size + f_i] = weight[((c * in_channel + c_in) * kernel_size * kernel_size) + ij];
        }
        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            const int i_mn = mn + reg_n * get_local_size(1);
            const int i_m = i_mn / N;
            const int i_n = i_mn % N;
            int input_index = ((c_in * M + i_m) * width_win) + ((i_n) * stride * kernel_size);
            for (int ij = 0; ij < kernel_size * kernel_size; ij++) {
                sum[reg_n] += weight_reg[ij] * input_win[input_index + ij];
            }
        }
    }

    for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
        output[(c * M + m) * N + n + reg_n * get_local_size(1)] = sum[reg_n] + bias[c];
    }
}

__kernel void im2win_channel_reg_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 2;
    const int reg_size_n = 1;
    const int local_size_c = get_local_size(0);
    const int local_size_mn = get_local_size(1);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int mn = get_group_id(1) * local_size_mn * reg_size_n + get_local_id(1);
    const int m = mn / N;
    const int n = mn % N;

    float sum[reg_size_c][reg_size_n];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_n; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        float weight_reg[reg_size_c][9];
        for (int w_c = 0; w_c < reg_size_c; w_c++) {
            for (int ij = 0; ij < kernel_size * kernel_size; ij++) {
                int f_i = ij / kernel_size;
                int f_j = ij % kernel_size;
                weight_reg[w_c][f_j * kernel_size + f_i] = weight[(((c + w_c * local_size_c) * in_channel + c_in) * kernel_size * kernel_size) + ij];
            }
        }

        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            const int i_mn = mn + reg_n * local_size_mn;
            const int i_m = i_mn / N;
            const int i_n = i_mn % N;
            int input_index = ((c_in * M + i_m) * width_win) + ((i_n) * stride * kernel_size);
            for (int ij = 0; ij < kernel_size * kernel_size; ij++) {
                for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                    sum[reg_c][reg_n] += weight_reg[reg_c][ij] * input_win[input_index + ij];
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            output[(c + reg_c * local_size_c) * M * N + (m * N + n + reg_n * local_size_mn)] = sum[reg_c][reg_n] + bias[c + reg_c * local_size_c];
        }
    }
}

__kernel void im2win_channel_reg_v4_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 16;
    const int reg_size_n = 1;
    const int local_size_c = get_local_size(0);
    const int local_size_mn = get_local_size(1);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int mn = get_group_id(1) * local_size_mn * reg_size_n + get_local_id(1);
    const int m = mn / N;
    const int n = mn % N;

    float sum[reg_size_c][reg_size_n];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_n; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            const int i_mn = mn + reg_n * local_size_mn;
            const int i_m = i_mn / N;
            const int i_n = i_mn % N;
            int input_index = ((c_in * M + i_m) * width_win) + ((i_n) * stride * kernel_size);
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                        int weight_index = ((((c + reg_c * local_size_c) * in_channel + c_in) * kernel_size) + j) * kernel_size + i;
                        sum[reg_c][reg_n] += weight[weight_index] * input_win[input_index + i * kernel_size + j];
                    }
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_n = 0; reg_n < reg_size_n; reg_n++) {
            output[(c + reg_c * local_size_c) * M * N + (m * N + n + reg_n * local_size_mn)] = sum[reg_c][reg_n] + bias[c + reg_c * local_size_c];
        }
    }
}

__kernel void im2win_channel_reg_transpose_v5_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 4;
    const int reg_size_m = 4;
    const int local_size_c = get_local_size(0);
    const int local_size_m = get_local_size(2);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int n = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int m = get_group_id(2) * local_size_m * reg_size_m + get_local_id(2);

    float sum[reg_size_c][reg_size_m];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_m; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                    int weight_index = ((((c + reg_c * local_size_c) * in_channel + c_in) * kernel_size) + j) * kernel_size + i;
                    float weight_tmp = weight[weight_index];
                    for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
                        int input_index = ((c_in * width_win + (n * stride * kernel_size) + i * kernel_size + j) * M) + (m + reg_m * local_size_m);
                        sum[reg_c][reg_m] += weight_tmp * input_win[input_index];
                    }
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
            output[(c + reg_c * local_size_c) * M * N + ((m + reg_m * local_size_m) * N + n)] = sum[reg_c][reg_m] + bias[c + reg_c * local_size_c];
        }
    }
}

__kernel void im2win_transpose(
    const int n,
    __global const float* data_im,
    const int data_im_off,
    const int height,
    const int width,
    const int kernel_size,
    const int padding,
    const int stride,
    const int height_win,
    const int width_win,
    __global float* data_win,
    const int data_win_off
) {
    // im = (4, 64, 64) kernel = (320, 4, 3, 3) padding = (1) stride = (1) => col = (64, 4*3*64)
    // n = channels * height_win * (width + 2*padding)
    // n x kernel_size
    const int width_pad = width + 2 * padding;
    const int index = get_global_id(0);
    const int h_index = index / width_pad;
    const int h_win = h_index % height_win;
    const int w_pad = index % width_pad;
    const int w_win = w_pad * kernel_size;
    const int c_im = h_index / height_win;
    const int h_offset = h_win * stride - padding;
    const int w_offset = w_pad - padding;

    // const int width_win = (width + 2 * padding) * kernel_size;
    __global float* data_win_ptr = data_win + data_win_off;
    data_win_ptr += (c_im * width_win + w_win) * height_win + h_win;

    __global const float* data_im_ptr = data_im + data_im_off;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

    for (int i = 0; i < kernel_size; ++i) {
        int h_im = h_offset + i;
        int w_im = w_offset;
        *data_win_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * width] : 0.0f;
        data_win_ptr += height_win;
    }
}

__kernel void im2win_transpose_reorder(
    const int n,
    __global const float* data_im,
    const int data_im_off,
    const int height,
    const int width,
    const int kernel_size,
    const int padding,
    const int stride,
    const int height_win,
    const int width_win,
    __global float* data_win,
    const int data_win_off
) {
    // im = (4, 64, 64) kernel = (320, 4, 3, 3) padding = (1) stride = (1) => col = (64, 4*3*64)
    // n = channels * height_win * (width + 2*padding)
    // n x kernel_size
    const int width_pad = width + 2 * padding;
    const int index = get_global_id(0);
    const int h_index = index / width_pad;
    const int h_win = h_index % height_win;
    const int w_pad = index % width_pad;
    const int w_win = w_pad * kernel_size;
    const int c_im = h_index / height_win;
    const int h_offset = h_win * stride - padding;
    const int w_offset = w_pad - padding;

    // const int width_win = (width + 2 * padding) * kernel_size;
    __global float* data_win_ptr = data_win + data_win_off;
    data_win_ptr += (c_im * width_win + w_pad) * height_win + h_win;

    __global const float* data_im_ptr = data_im + data_im_off;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

    for (int i = 0; i < kernel_size; ++i) {
        int h_im = h_offset + i;
        int w_im = w_offset;
        *data_win_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * width] : 0.0f;
        data_win_ptr += width_pad * height_win;
    }
}

__kernel void im2win_channel_reg_transpose_vector_v6_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const floatX *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 4;
    const int reg_size_m = 4;
    const int local_size_c = get_local_size(0);
    const int local_size_m = get_local_size(2);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int n = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int m = get_group_id(2) * local_size_m * reg_size_m + get_local_id(2) * reg_size_m;

    float sum[reg_size_c][reg_size_m];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_m; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                    int weight_index = ((((c + reg_c * local_size_c) * in_channel + c_in) * kernel_size) + j) * kernel_size + i;
                    float weight_tmp = weight[weight_index];
                    for (int reg_m = 0; reg_m < reg_size_m / WIDTH; reg_m++) {
                        int input_index = ((c_in * width_win + (n * stride * kernel_size) + i * kernel_size + j) * M) + (m + reg_m * WIDTH);
                        floatX input_tmp = input_win[input_index / WIDTH];
#if WIDTH == 1
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp;
#elif WIDTH == 2
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.x;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.y;
#elif WIDTH == 4
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.x;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.y;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.z;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.w;
#elif WIDTH == 8
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.s0;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.s1;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.s2;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.s3;
                        sum[reg_c][reg_m * WIDTH + 4] += weight_tmp * input_tmp.s4;
                        sum[reg_c][reg_m * WIDTH + 5] += weight_tmp * input_tmp.s5;
                        sum[reg_c][reg_m * WIDTH + 6] += weight_tmp * input_tmp.s6;
                        sum[reg_c][reg_m * WIDTH + 7] += weight_tmp * input_tmp.s7;
#elif WIDTH == 16
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.s0;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.s1;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.s2;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.s3;
                        sum[reg_c][reg_m * WIDTH + 4] += weight_tmp * input_tmp.s4;
                        sum[reg_c][reg_m * WIDTH + 5] += weight_tmp * input_tmp.s5;
                        sum[reg_c][reg_m * WIDTH + 6] += weight_tmp * input_tmp.s6;
                        sum[reg_c][reg_m * WIDTH + 7] += weight_tmp * input_tmp.s7;
                        sum[reg_c][reg_m * WIDTH + 8] += weight_tmp * input_tmp.s8;
                        sum[reg_c][reg_m * WIDTH + 9] += weight_tmp * input_tmp.s9;
                        sum[reg_c][reg_m * WIDTH + 10] += weight_tmp * input_tmp.sA;
                        sum[reg_c][reg_m * WIDTH + 11] += weight_tmp * input_tmp.sB;
                        sum[reg_c][reg_m * WIDTH + 12] += weight_tmp * input_tmp.sC;
                        sum[reg_c][reg_m * WIDTH + 13] += weight_tmp * input_tmp.sD;
                        sum[reg_c][reg_m * WIDTH + 14] += weight_tmp * input_tmp.sE;
                        sum[reg_c][reg_m * WIDTH + 15] += weight_tmp * input_tmp.sF;
#endif
                    }
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
            output[(c + reg_c * local_size_c) * M * N + ((m + reg_m) * N + n)] = sum[reg_c][reg_m] + bias[c + reg_c * local_size_c];
        }
    }
}

__kernel void im2win_channel_reg_transpose_weight_vector_v7_matmul(
    __global const floatX *weight,
    __global const float *bias,
    __global const float *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 1;
    const int reg_size_m = 8;
    const int local_size_c = get_local_size(0);
    const int local_size_m = get_local_size(2);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int n = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int m = get_group_id(2) * local_size_m * reg_size_m + get_local_id(2);

    float sum[reg_size_c][reg_size_m];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_m; j++) {
            sum[i][j] = 0.0f;
        }
    }

    floatX weight_tmp[reg_size_c];
    for (int c_in = 0; c_in < in_channel; c_in++) {
        for (int j = 0; j < kernel_size; j++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                    int weight_index = ((((c + reg_c * local_size_c) * in_channel + c_in) * kernel_size) + j) * kernel_size + i;
                    if ((i == 0 && j == 0) || weight_index % WIDTH == 0) {
                        weight_tmp[reg_c] = weight[weight_index/ WIDTH];
                    }
                    for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
                        int input_index = ((c_in * width_win + (n * stride * kernel_size) + i * kernel_size + j) * M) + (m + reg_m * local_size_m);
                        #if WIDTH == 1
                        sum[reg_c][reg_m] += weight_tmp[reg_c] * input_win[input_index];
                        #elif WIDTH == 2
                        switch (weight_index % WIDTH) {
                            case 0:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].x * input_win[input_index];
                                break;
                            case 1:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].y * input_win[input_index];
                                break;
                        }
                        #elif WIDTH == 4
                        switch (weight_index % WIDTH) {
                            case 0:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].x * input_win[input_index];
                                break;
                            case 1:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].y * input_win[input_index];
                                break;
                            case 2:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].z * input_win[input_index];
                                break;
                            case 3:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].w * input_win[input_index];
                                break;
                        }
                        #elif WIDTH == 8
                        switch (weight_index % WIDTH) {
                            case 0:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s0 * input_win[input_index];
                                break;
                            case 1:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s1 * input_win[input_index];
                                break;
                            case 2:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s2 * input_win[input_index];
                                break;
                            case 3:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s3 * input_win[input_index];
                                break;
                            case 4:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s4 * input_win[input_index];
                                break;
                            case 5:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s5 * input_win[input_index];
                                break;
                            case 6:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s6 * input_win[input_index];
                                break;
                            case 7:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s7 * input_win[input_index];
                                break;
                        }
                        #elif WIDTH == 16
                        switch (weight_index % WIDTH) {
                            case 0:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s0 * input_win[input_index];
                                break;
                            case 1:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s1 * input_win[input_index];
                                break;
                            case 2:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s2 * input_win[input_index];
                                break;
                            case 3:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s3 * input_win[input_index];
                                break;
                            case 4:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s4 * input_win[input_index];
                                break;
                            case 5:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s5 * input_win[input_index];
                                break;
                            case 6:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s6 * input_win[input_index];
                                break;
                            case 7:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s7 * input_win[input_index];
                                break;
                            case 8:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s8 * input_win[input_index];
                                break;
                            case 9:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].s9 * input_win[input_index];
                                break;
                            case 10:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sA * input_win[input_index];
                                break;
                            case 11:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sB * input_win[input_index];
                                break;
                            case 12:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sC * input_win[input_index];
                                break;
                            case 13:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sD * input_win[input_index];
                                break;
                            case 14:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sE * input_win[input_index];
                                break;
                            case 15:
                                sum[reg_c][reg_m] += weight_tmp[reg_c].sF * input_win[input_index];
                                break;
                        }
                        #endif
                    }
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
            output[(c + reg_c * local_size_c) * M * N + ((m + reg_m * local_size_m) * N + n)] = sum[reg_c][reg_m] + bias[c + reg_c * local_size_c];
        }
    }
}

__kernel void im2win_channel_reg_transpose_reorder_vector_v8_matmul(
    __global const float *weight,
    __global const float *bias,
    __global const floatX *input_win,
    __global float *output,
    const int M,
    const int N,
    const int width_win,
    const int in_channel,
    const int kernel_size,
    const int stride
) {
    const int reg_size_c = 4;
    const int reg_size_m = 4;
    const int local_size_c = get_local_size(0);
    const int local_size_m = get_local_size(2);
    const int c = get_group_id(0) * local_size_c * reg_size_c + get_local_id(0);
    const int n = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int m = get_group_id(2) * local_size_m * reg_size_m + get_local_id(2) * reg_size_m;
    const int win_pad = width_win / kernel_size;

    float sum[reg_size_c][reg_size_m];
    for (int i = 0; i < reg_size_c; i++) {
        for (int j = 0; j < reg_size_m; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int c_in = 0; c_in < in_channel; c_in++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
                    int weight_index = ((((c + reg_c * local_size_c) * in_channel + c_in) * kernel_size) + j) * kernel_size + i;
                    float weight_tmp = weight[weight_index];
                    for (int reg_m = 0; reg_m < reg_size_m / WIDTH; reg_m++) {
//                        int input_index = ((c_in * width_win + (n * stride * kernel_size) + i * kernel_size + j) * M) + (m + reg_m * WIDTH);
                        int input_index = ((c_in * width_win + (n * stride) + i + j * win_pad) * M) + (m + reg_m * WIDTH);
                        floatX input_tmp = input_win[input_index / WIDTH];
#if WIDTH == 1
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp;
#elif WIDTH == 2
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.x;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.y;
#elif WIDTH == 4
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.x;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.y;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.z;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.w;
#elif WIDTH == 8
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.s0;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.s1;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.s2;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.s3;
                        sum[reg_c][reg_m * WIDTH + 4] += weight_tmp * input_tmp.s4;
                        sum[reg_c][reg_m * WIDTH + 5] += weight_tmp * input_tmp.s5;
                        sum[reg_c][reg_m * WIDTH + 6] += weight_tmp * input_tmp.s6;
                        sum[reg_c][reg_m * WIDTH + 7] += weight_tmp * input_tmp.s7;
#elif WIDTH == 16
                        sum[reg_c][reg_m * WIDTH] += weight_tmp * input_tmp.s0;
                        sum[reg_c][reg_m * WIDTH + 1] += weight_tmp * input_tmp.s1;
                        sum[reg_c][reg_m * WIDTH + 2] += weight_tmp * input_tmp.s2;
                        sum[reg_c][reg_m * WIDTH + 3] += weight_tmp * input_tmp.s3;
                        sum[reg_c][reg_m * WIDTH + 4] += weight_tmp * input_tmp.s4;
                        sum[reg_c][reg_m * WIDTH + 5] += weight_tmp * input_tmp.s5;
                        sum[reg_c][reg_m * WIDTH + 6] += weight_tmp * input_tmp.s6;
                        sum[reg_c][reg_m * WIDTH + 7] += weight_tmp * input_tmp.s7;
                        sum[reg_c][reg_m * WIDTH + 8] += weight_tmp * input_tmp.s8;
                        sum[reg_c][reg_m * WIDTH + 9] += weight_tmp * input_tmp.s9;
                        sum[reg_c][reg_m * WIDTH + 10] += weight_tmp * input_tmp.sA;
                        sum[reg_c][reg_m * WIDTH + 11] += weight_tmp * input_tmp.sB;
                        sum[reg_c][reg_m * WIDTH + 12] += weight_tmp * input_tmp.sC;
                        sum[reg_c][reg_m * WIDTH + 13] += weight_tmp * input_tmp.sD;
                        sum[reg_c][reg_m * WIDTH + 14] += weight_tmp * input_tmp.sE;
                        sum[reg_c][reg_m * WIDTH + 15] += weight_tmp * input_tmp.sF;
#endif
                    }
                }
            }
        }
    }

    for (int reg_c = 0; reg_c < reg_size_c; reg_c++) {
        for (int reg_m = 0; reg_m < reg_size_m; reg_m++) {
            output[(c + reg_c * local_size_c) * M * N + ((m + reg_m) * N + n)] = sum[reg_c][reg_m] + bias[c + reg_c * local_size_c];
        }
    }
}