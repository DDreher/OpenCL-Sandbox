#include "HashTable/HashTable.h"
#include <Base\OpenCLManager.h>
#include "assert.h"
#include "Base\Definitions.h"
#include "Base\Utilities.h"

HashTable::HashTable(uint32_t size) : size_(size)
{
    Init(size);

    /* initialize random seed: */
    srand(42);
    
    params.resize(NUM_PARAMS);
    GenerateParams();
}

HashTable::~HashTable()
{
    // Release buffers
    cl_int status = 0;
    status = clReleaseMemObject(table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
}

void HashTable::Init(uint32_t table_size)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // Allocate memory for table and initialize with empty values
    //std::vector<Entry> cuckoo(table_size);
    //std::fill(cuckoo.begin(), cuckoo.end(), HashTable::EMPTY);

    // 1. Allocate enough memory on GPU to fit hash table, padded to wavefront size
    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(table_size), static_cast<uint32_t>(mpp::constants::WAVEFRONT_SIZE)));
    table_buffer_ = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint64_t), NULL, NULL);

    // 2. Initialize all the memory with empty elements
    // Todo: Merge with step 3 possible?
    std::vector<uint64_t> empty_elements(next_multiple);
    std::fill(empty_elements.begin(), empty_elements.end(), mpp::constants::EMPTY);
    status = clEnqueueWriteBuffer(mgr->command_queue, table_buffer_, CL_TRUE, 0, next_multiple * sizeof(uint64_t), empty_elements.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    //// If necessary pad to multiple of wavefront size
    //if (elements.size() < next_multiple)
    //{
    //    size_t offset = elements.size() * sizeof(uint64_t);
    //    size_t num_bytes_written = (next_multiple - elements.size()) * sizeof(uint64_t);
    //    status = clEnqueueWriteBuffer(mgr->command_queue, table_buffer_, CL_TRUE, offset, num_bytes_written, &mpp::constants::EMPTY, 0, NULL, NULL);
    //    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    //}

    // 3. Insert the initial elements into the hash table
    //bool insert_succesful = Insert(elements);

    // 4. If insertion failed we have to reconstruct the table with new parameters
    // TODO: Reconstruct hash table if insert failed
}

bool HashTable::Insert(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values)
{
    assert(keys.size() == values.size());

    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // 1. Allocate GPU memory for key-val-pairs to insert, padded to size of wavefront
    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(keys.size()), static_cast<uint32_t>(mpp::constants::WAVEFRONT_SIZE)));

    cl_mem keys_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint32_t), NULL, NULL);
    cl_mem values_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint32_t), NULL, NULL);

    // 2. Fill buffers
    status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, 0, keys.size() * sizeof(uint32_t), keys.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clEnqueueWriteBuffer(mgr->command_queue, values_buffer, CL_TRUE, 0, values.size() * sizeof(uint32_t), values.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // If necessary write padding
    if (keys.size() < next_multiple)
    {
        uint32_t num_padded_elements = next_multiple - static_cast<uint32_t>(keys.size());
        std::vector<uint32_t> empty_elements(num_padded_elements);
        std::fill(empty_elements.begin(), empty_elements.end(), mpp::constants::EMPTY);
        size_t offset = keys.size() * sizeof(uint32_t);
        size_t num_bytes_written = num_padded_elements * sizeof(uint32_t);
        status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clEnqueueWriteBuffer(mgr->command_queue, values_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // 3. Construct parameters buffer
    cl_mem params_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, params.size() * sizeof(uint32_t), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, params_buffer, CL_TRUE, 0, params.size() * sizeof(uint32_t), params.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 4. Construct status buffer
    cl_mem status_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, NULL);
    int intital_status = 0;
    status = clEnqueueWriteBuffer(mgr->command_queue, status_buffer, CL_TRUE, 0, sizeof(uint32_t), &intital_status, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 5. Run kernel    
    const cl_kernel kernel_hashtable_insert = mgr->kernel_map[mpp::kernels::HASHTABLE_INSERT];
    // args: __global int64_t* key_val_pairs, __global int64_t* table, __constant uint32_t* params, __global uint8_t* out_status
    status = clSetKernelArg(kernel_hashtable_insert, 0, sizeof(cl_mem), (void*)&keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 1, sizeof(cl_mem), (void*)&values_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 2, sizeof(cl_mem), (void*)&table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 3, sizeof(cl_mem), (void*)&params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 4, sizeof(cl_mem), (void*)&status_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    size_t global_work_size[1] = { static_cast<size_t>(next_multiple) };
    size_t local_work_size[1] = { std::min(static_cast<size_t>(THREAD_BLOCK_SIZE), static_cast<size_t>(next_multiple)) };
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_hashtable_insert, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // TODO: Error checking (collisions) -> Reconstruction

    // 6. Cleanup -> Release buffers
    status = clReleaseMemObject(keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(values_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(status_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    
    return status == mpp::ReturnCode::CODE_SUCCESS;
}

std::vector<uint32_t> HashTable::Get(const std::vector<uint32_t>& keys)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // 1. Allocate GPU memory
    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(keys.size()), static_cast<uint32_t>(mpp::constants::WAVEFRONT_SIZE)));
    cl_mem key_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint32_t), NULL, NULL);

    // 2. Fill buffer
    status = clEnqueueWriteBuffer(mgr->command_queue, key_buffer, CL_TRUE, 0, keys.size() * sizeof(uint32_t), keys.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // If necessary write padding
    if (keys.size() < next_multiple)
    {
        uint32_t num_padded_elements = next_multiple - static_cast<uint32_t>(keys.size());
        std::vector<uint32_t> empty_elements(num_padded_elements);
        std::fill(empty_elements.begin(), empty_elements.end(), mpp::constants::EMPTY);
        size_t offset = keys.size() * sizeof(uint32_t);
        size_t num_bytes_written = num_padded_elements * sizeof(uint32_t);
        status = clEnqueueWriteBuffer(mgr->command_queue, key_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // 3. Construct parameters buffer
    cl_mem params_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, params.size() * sizeof(uint32_t), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, params_buffer, CL_TRUE, 0, params.size() * sizeof(uint32_t), params.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 4. invoke retrieve kernel
    const cl_kernel kernel_hashtable_retrieve = mgr->kernel_map[mpp::kernels::HASHTABLE_RETRIEVE];
    // params: __global int32_t* keys, __global int64_t* table, __constant uint32_t* params
    status = clSetKernelArg(kernel_hashtable_retrieve, 0, sizeof(cl_mem), (void*)&key_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_retrieve, 1, sizeof(cl_mem), (void*)&table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_retrieve, 2, sizeof(cl_mem), (void*)&params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    size_t global_work_size[1] = { static_cast<size_t>(next_multiple) };
    size_t local_work_size[1] = { std::min(static_cast<size_t>(THREAD_BLOCK_SIZE), static_cast<size_t>(next_multiple)) };

    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_hashtable_retrieve, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 5. Read back results
    std::vector<uint32_t> retrieved_entries(next_multiple);
    status = clEnqueueReadBuffer(mgr->command_queue, key_buffer, CL_TRUE, 0, next_multiple * sizeof(uint32_t), retrieved_entries.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 6. Release buffers
    status = clReleaseMemObject(key_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    retrieved_entries.resize(keys.size());
    return retrieved_entries;
}

void HashTable::GenerateParams()
{
    params.clear();

    //PARAM_IDX_HASHFUNC_A_0   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_B_0   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_A_1   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_B_1   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_A_2   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_B_2   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_A_3   
    params.push_back(rand());

    //PARAM_IDX_HASHFUNC_B_3   
    params.push_back(rand());

    //PARAM_IDX_MAX_ITERATIONS 
    params.push_back(max_iterations);
    
    //PARAM_IDX_TABLESIZE      
    params.push_back(size_);
}
