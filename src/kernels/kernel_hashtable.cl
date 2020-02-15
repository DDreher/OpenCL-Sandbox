#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

// Typedefs for better comparison of host and device types
typedef char				int8_t;
typedef unsigned char		uint8_t;
typedef short				int16_t;
typedef unsigned short		uint16_t;
typedef int					int32_t;
typedef unsigned int		uint32_t;
typedef long				int64_t;
typedef unsigned long		uint64_t;

#define STATUS_SUCCESS 0
#define STATUS_ERROR 1
#define HASH_P 334214459 
#define KEY_EMPTY 0xFFFFFFFF

#define PARAM_IDX_HASH_FUNC_A_0		0
#define PARAM_IDX_HASH_FUNC_B_0		1
#define PARAM_IDX_HASH_FUNC_A_1		2
#define PARAM_IDX_HASH_FUNC_B_1		3
#define PARAM_IDX_HASH_FUNC_A_2		4
#define PARAM_IDX_HASH_FUNC_B_2		5
#define PARAM_IDX_HASH_FUNC_A_3		6
#define PARAM_IDX_HASH_FUNC_B_3		7
#define PARAM_IDX_MAX_ITERATIONS	8
#define PARAM_IDX_TABLESIZE			9

#define GET_KEY(entry) ( (uint32_t)((entry) >> 32) )
#define GET_VALUE(entry) ((uint32_t)((entry)))
#define MAKE_ENTRY(key,value) ( (((uint64_t)key) << 32) + (value) )
#define HASH_FUNCTION(key, a, b, table_size) ( (a * key + b) % HASH_P % table_size  )

__kernel void Insert(__global uint32_t* keys, __global uint32_t* values, __global uint64_t* table, __constant uint32_t* params, __global uint32_t* status)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	// Load up the key value pair into a 64 bit int.
	uint32_t key = keys[global_id];
	uint32_t value = values[global_id];
	uint64_t entry = MAKE_ENTRY(key, value);

	// DEBUG
	//if (key != GET_KEY(entry))
	//{
	//	printf("%s\n", "ASSERT KEY CHECK FAILED\n");
	//}
	//if (value != GET_VALUE(entry))
	//{
	//	printf("%s\n", "ASSERT KEY CHECK FAILED\n");
	//}

	if (key == KEY_EMPTY)
	{
		// KEY_EMPTY is reserved and used for padded values -> We just return
		keys[global_id] = STATUS_SUCCESS;
		return;
	}

	// New items are always inserted using their first hash function.
	uint32_t location = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_0], params[PARAM_IDX_HASH_FUNC_B_0], params[PARAM_IDX_TABLESIZE]);

	// Repeat the insertion process while the thread still has an item.
	for (uint32_t i = 0; i < params[PARAM_IDX_MAX_ITERATIONS]; ++i)
	{
		// Insert the new item and check for an eviction.
		entry = atomic_xchg(&table[location], entry);
		key = GET_KEY(entry);

		if (key == KEY_EMPTY) 
		{
			keys[global_id] = STATUS_SUCCESS;
			return;
		}
	
		// If an item was evicted, figure out where to reinsert the entry.
		uint32_t location_0 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_0], params[PARAM_IDX_HASH_FUNC_B_0], params[PARAM_IDX_TABLESIZE]);
		uint32_t location_1 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_1], params[PARAM_IDX_HASH_FUNC_B_1], params[PARAM_IDX_TABLESIZE]);
		uint32_t location_2 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_2], params[PARAM_IDX_HASH_FUNC_B_2], params[PARAM_IDX_TABLESIZE]);
		uint32_t location_3 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_3], params[PARAM_IDX_HASH_FUNC_B_3], params[PARAM_IDX_TABLESIZE]);

		// Cycle through hash functions (round robin fashion)
		if (location == location_0) 
		{
			location = location_1;
		}
		else if (location == location_1)
		{
			location = location_2;
		}
		else if (location == location_2) 
		{
			location = location_3;
		}
		else
		{
			location = location_0;
		}
	}

	// The eviction chain was too long; report the failure.
	keys[global_id] = STATUS_ERROR;
	status[0] |= STATUS_ERROR;
}

__kernel void Retrieve(__global int32_t* keys, __global int32_t* out_values, __global int64_t* table, __constant uint32_t* params)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	uint32_t key = keys[global_id];
	if (key == KEY_EMPTY)
	{
		return;
	}

	// Cycle through all potential locations in hash table and check if requested key exists
	uint32_t location_0 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_0], params[PARAM_IDX_HASH_FUNC_B_0], params[PARAM_IDX_TABLESIZE]);
	uint32_t location_1 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_1], params[PARAM_IDX_HASH_FUNC_B_1], params[PARAM_IDX_TABLESIZE]);
	uint32_t location_2 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_2], params[PARAM_IDX_HASH_FUNC_B_2], params[PARAM_IDX_TABLESIZE]);
	uint32_t location_3 = HASH_FUNCTION(key, params[PARAM_IDX_HASH_FUNC_A_3], params[PARAM_IDX_HASH_FUNC_B_3], params[PARAM_IDX_TABLESIZE]);

	uint64_t entry = 0;

	if (GET_KEY(entry = table[location_0]) != key)
	{
		if (GET_KEY(entry = table[location_1]) != key)
		{
			if (GET_KEY(entry = table[location_2]) != key)
			{
				if (GET_KEY(entry = table[location_3]) != key)
				{
					// Did not find the requested key
					out_values[global_id] = KEY_EMPTY;
				}
			}
		}
	}

	out_values[global_id] = GET_VALUE(entry);
}
