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

#define GET_KEY(entry) ( (uint32_t)((entry) >> 32) )
#define MAKE_ENTRY(key,value) ( (((uint64_t)key) << 32) + (value) )
#define HASH_FUNCTION(key, a, b, table_size) ( (a * key + b) % HASH_P % table_size  )

__kernel void Insert(__global int32_t* keys, __global int32_t* values, __global int32_t* table, __private uint32_t num_elements, __constant uint32_t* ptr_max_iterations,
	__constant uint32_t* hash_constants, __constant uint32_t* ptr_table_size, __global uint8_t* out_status)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	uint32_t max_iterations = *ptr_max_iterations;
	uint32_t table_size = *ptr_table_size;

	// Load up the key value pair into a 64 bit entry.
	uint32_t key = keys[local_id];
	uint32_t value = values[local_id];
	uint64_t entry = MAKE_ENTRY(key, value);

	// New items are always inserted using their first hash function.
	uint32_t location = HASH_FUNCTION(key, hash_constants[0], hash_constants[1], table_size);

	// Repeat the insertion process while the thread still has an item.
	for (uint32_t i = 0; i < max_iterations; ++i)
	{
		// Insert the new item and check for an eviction.
		entry = atomic_xchg(&table[location], entry);
		key = GET_KEY(entry);
	
		if (key == KEY_EMPTY) 
		{
			out_status[global_id] = STATUS_SUCCESS;
			return;
		}
	
		// If an item was evicted, figure out where to reinsert the entry.
		uint32_t location_1 = HASH_FUNCTION(key, hash_constants[0], hash_constants[2], table_size);
		uint32_t location_2 = HASH_FUNCTION(key, hash_constants[2], hash_constants[3], table_size);
		uint32_t location_3 = HASH_FUNCTION(key, hash_constants[4], hash_constants[5], table_size);
		uint32_t location_4 = HASH_FUNCTION(key, hash_constants[6], hash_constants[7], table_size);

		// Cycle through hash functions (round robin fashion)
		if (location == location_1) 
		{
			location = location_2;
		}
		else if (location == location_2)
		{
			location = location_3;
		}
		else if (location == location_3) 
		{
			location = location_4;
		}
		else
		{
			location = location_1;
		}
	}

	// The eviction chain was too long; report the failure.
	out_status[global_id] = STATUS_ERROR;
}

__kernel void Retrieve(__global int32_t* buffer_a, __global int32_t* buffer_b, __global int32_t* buffer_c)
{
	int32_t global_id = get_global_id(0);
	int32_t local_id = get_local_id(0);
	int32_t group_id = get_group_id(0);

	// TODO
}




