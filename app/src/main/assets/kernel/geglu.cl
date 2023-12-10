__kernel void gelu_multiply(
    __global float *AB,
    __global float *dst
) {
    int offset = get_global_size(1);
    int idx = get_global_id(0) * offset * 2 + get_global_id(1);
    const float a = AB[idx];
    const float b = AB[idx + offset];

    // GELU(x) = x * 0.5 * (1.0 + erf( x / \sqrt{2}))
    // tanh version
    // const float x = src[idx];
    // float temp = pown(x, 3);
    // temp = fma(temp, 0.044715f, x);
    // temp *= sqrt(M_2_PI_F);
    // temp = tanh(temp) + 1.0f;
    // dst[idx] = 0.5f * x * temp;

    float temp = b * M_SQRT1_2_F;
    temp = 1.0f + erf(temp);
    temp = b * 0.5f * temp;

    int dst_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    dst[dst_idx] = a * temp;
}