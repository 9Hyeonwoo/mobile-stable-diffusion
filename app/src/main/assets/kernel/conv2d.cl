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