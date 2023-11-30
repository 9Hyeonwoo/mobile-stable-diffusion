/*
 * local_reduction_mean
 * max diff: 0.00000000017462298274. error occurs.
 */
__kernel void local_reduction_mean(__global const float *input,
                        __global float *output,
                        __local float* reductionSums)
{
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = get_group_id(0);

    const int i = workgroupID * localSize * 4 + localID;
	reductionSums[localID] = input[i] + input[i + localSize] + input[i + localSize * 2] + input[i + localSize * 3];

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {	// the root of the reduction subtree
		output[workgroupID] = reductionSums[0] / (localSize * 4) ;
	}
}

__kernel void local_reduction_variance(__global const float *input,
                        __global const float *mean,
                        __global float *output,
                        __local float* reductionSums)
{
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = get_group_id(0);

    const int i = workgroupID * localSize * 4 + localID;
    reductionSums[localID] = 0;
    for (int j = 0; j < 4; j++) {
        reductionSums[localID] += (input[i + localSize * j] - mean[workgroupID]) * (input[i + localSize * j] - mean[workgroupID]);
    }

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {
	    // localSize * 4 - 1 : bias=True.
		output[workgroupID] = reductionSums[0] / (localSize * 4);
	}
}

__constant float epsilon = 0.00001;

/* layer_norm
 * max diff: 0.00000381469726562500
 */
__kernel void layer_norm(__global const float *input,
                        __global const float *mean,
                        __global const float *variance,
                        __global const float *weight,
                        __global const float *bias,
                        const size_t chunkSize,
                        __global float *output
) {
    const int globalID = get_global_id(0);
    const int chunkID = globalID % chunkSize;
    const int chunkGroupID = globalID / chunkSize;

    float temp = input[globalID] - mean[chunkGroupID];
    temp /= sqrt(variance[chunkGroupID] + epsilon);
    output[globalID] = fma(temp, weight[chunkID], bias[chunkID]);
}