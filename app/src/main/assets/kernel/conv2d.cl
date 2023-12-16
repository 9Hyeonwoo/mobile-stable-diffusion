__kernel void conv2d(__global float *input,
                     __global float *weight,
                     __global float *bias,
                     __global float *output,
                     const size_t input_size,
                     const size_t kernel_channel_size,
                     const size_t kernel_size,
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
    const size_t n,
    __global const float* data_im,
    const int data_im_off,
    const size_t height,
    const size_t width,
    const size_t kernel_size,
    const int padding,
    const int stride,
    const size_t height_col,
    const size_t width_col,
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
    const size_t M,
    const size_t N,
    const size_t K
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
    const size_t n,
    __global const float* data_im,
    const int data_im_off,
    const size_t height,
    const size_t width,
    const size_t kernel_size,
    const int padding,
    const int stride,
    const size_t height_win,
    const size_t width_win,
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
    const size_t C,
    const size_t M,
    const size_t N,
    const size_t width_win,
    const size_t in_channel,
    const size_t kernel_size,
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