__kernel void up_sample_nearest(
    __global float *input,
    __global float *output,
    const size_t scale
) {
    int channel = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);
    int inputHeight = get_global_size(1);
    int input_index = (channel * inputHeight * inputHeight) + (i * inputHeight) + j;
    float input_value = input[input_index];

    for (int i_offset = 0; i_offset < scale; i_offset++) {
        for (int j_offset = 0; j_offset < scale; j_offset++) {
            int output_index = (channel * scale * inputHeight * scale * inputHeight) + ((i * scale + i_offset) * scale * inputHeight) + (j * scale + j_offset);
            output[output_index] = input_value;
        }
    }
}