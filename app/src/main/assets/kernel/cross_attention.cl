__kernel void einsum_bik_bjk_bij(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const size_t kSize,
    const float scale
) {
    int b = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    int iSize = get_global_size(1);
    int jSize = get_global_size(2);

    float sum = 0.0f;
    for (int k = 0; k < kSize; k++) {
        sum += A[(b * iSize * kSize) + (i * kSize) + k] * B[(b * jSize * kSize) + (j * kSize) + k];
    }

    C[(b * iSize * jSize) + (i * jSize) + j] = sum * scale;
}

__kernel void einsum_bij_bjk_bik(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const size_t jSize
) {
    int b = get_global_id(0);
    int i = get_global_id(1);
    int k = get_global_id(2);

    int iSize = get_global_size(1);
    int kSize = get_global_size(2);

    float sum = 0.0f;
    for (int j = 0; j < jSize; j++) {
        sum += A[(b * iSize * jSize) + (i * jSize) + j] * B[(b * jSize * kSize) + (j * kSize) + k];
    }

    C[(b * iSize * kSize) + (i * kSize) + k] = sum;
}
