__kernel void local_mean(__global const float *input, __global float *output, const size_t chunkSize, __global float *temp, __local float* reductionSums) {
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;
	const int chunkGroupID = globalID / chunkSize;
	const int multiplier = chunkSize / localSize;
	const int IDinMean = workgroupID % multiplier;

	reductionSums[localID] = input[globalID];

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {	// the root of the reduction subtree
		temp[workgroupID] = reductionSums[0];
	}

	for (int offset = multiplier / 2; offset > 0; offset /= 2) {
        barrier(CLK_GLOBAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
        if(localID == 0 && IDinMean < offset) {
            temp[workgroupID] += temp[workgroupID + offset];
        }
    }
    if (localID == 0 && IDinMean == 0) {
        output[chunkGroupID] = temp[workgroupID] / chunkSize;
    }
}

__kernel void local_variance(__global const float *input, __global const float *mean, __global float *output, const size_t chunkSize, __global float *temp, __local float* reductionSums) {
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;
	const int chunkGroupID = globalID / chunkSize;
	const int multiplier = chunkSize / localSize;
	const int IDinMean = workgroupID % multiplier;

	reductionSums[localID] = (input[globalID] - mean[chunkGroupID]) * (input[globalID] - mean[chunkGroupID]);

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {	// the root of the reduction subtree
		temp[workgroupID] = reductionSums[0];
	}

	for (int offset = multiplier / 2; offset > 0; offset /= 2) {
        barrier(CLK_GLOBAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
        if(localID == 0 && IDinMean < offset) {
            temp[workgroupID] += temp[workgroupID + offset];
        }
    }
    if (localID == 0 && IDinMean == 0) {
        output[chunkGroupID] = temp[workgroupID] / chunkSize;
    }
}

__kernel void _local_mean(__global const float *input, __global float *output, __local float* reductionSums) {
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;

	reductionSums[localID] = input[globalID];

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {	// the root of the reduction subtree
		output[workgroupID] = reductionSums[0] / localSize;
	}
}

__kernel void _local_variance(__global const float *input, __global const float *mean, __global float *output, __local float* reductionSums) {
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;

	reductionSums[localID] = (input[globalID] - mean[workgroupID]) * (input[globalID] - mean[workgroupID]);

	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}

	if(localID == 0) {	// the root of the reduction subtree
		output[workgroupID] = reductionSums[0] / localSize;
	}
}

__constant float epsilon = 1e-5;

__kernel void layer_norm(__global const float *input, __global const float *mean, __global const float *variance, __global const float *weight, __global const float *bias, const size_t chunkSize, __global float *output) {
    const int globalID = get_global_id(0);
    const int chunkID = globalID % chunkSize;
    const int chunkGroupID = globalID / chunkSize;

    const float temp = (input[globalID] - mean[chunkGroupID]) / sqrt(variance[chunkGroupID] + epsilon);
    output[globalID] = temp * weight[chunkID] + bias[chunkID];
}