#define MAX_THREADS_PER_CU 256
#define LOG2_MAX_THREADS_PER_CU 8

// Typedefs for better comparison of host and device types
typedef char				int8_t;
typedef unsigned char		uint8_t;
typedef short				int16_t;
typedef unsigned short		uint16_t;
typedef int					int32_t;
typedef unsigned int		uint32_t;
typedef long				int64_t;
typedef unsigned long		uint64_t;

__kernel void PrefixSum256(__global int32_t* buffer_a, __global int32_t* buffer_b, __global int32_t* buffer_c)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	__local int32_t local_array[MAX_THREADS_PER_CU];

	// copy to local memory
	local_array[local_id] = buffer_a[global_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	int32_t tree_depth = LOG2_MAX_THREADS_PER_CU; // Depth of a balanced tree with k leaves is log(k)
	size_t depth = 0;

	// Up-Sweep / Reduce Phase
	int32_t num_working_items = MAX_THREADS_PER_CU >> 1;
	int32_t offset = 1;
	for (size_t depth = 0; depth<tree_depth; ++depth)
	{
		if (local_id < num_working_items) 
		{
			size_t index_1 = local_id * (offset << 1) + offset - 1;
			size_t index_2 = index_1 + offset;
			local_array[index_2] = local_array[index_1] + local_array[index_2];
		}

		offset <<= 1;
		num_working_items >>= 1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Down-Sweep Phase
	if (local_id == MAX_THREADS_PER_CU -1)
	{
		local_array[MAX_THREADS_PER_CU -1] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	num_working_items = 1;
	offset = MAX_THREADS_PER_CU >> 1;
	for (size_t depth = 0; depth<tree_depth; ++depth)
	{
		if (local_id < num_working_items)
		{
			size_t index_1 = local_id*(offset << 1) + offset - 1;
			size_t index_2 = index_1 + offset;
			int32_t tmp = local_array[index_1];
			local_array[index_1] = local_array[index_2];
			local_array[index_2] = tmp + local_array[index_2];
		}

		num_working_items <<= 1;
		offset >>= 1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write resulting buffer_b
	buffer_b[global_id] = local_array[local_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// write resulting buffer_c
	if(local_id == MAX_THREADS_PER_CU -1)
	{
		buffer_c[group_id] = buffer_a[global_id] + buffer_b[global_id];
	}
}

__kernel void CalcE(__global int32_t* buffer_b, __global int32_t* buffer_d, __global int32_t* buffer_e, __private uint32_t num_elements)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	if (global_id < num_elements)
	{
		buffer_e[global_id] = buffer_b[global_id] + buffer_d[group_id];
	}
}
