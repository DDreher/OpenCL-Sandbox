#include "HashTable/HashTable.h"
#include <Base\OpenCLManager.h>
#include "assert.h"
#include "Base\Definitions.h"
#include "Base\Utilities.h"

HashTable::HashTable(size_t size) : size_(size)
{
    Init(size);
}

HashTable::~HashTable()
{
    // Release buffers
    cl_int status = 0;
    status = clReleaseMemObject(table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
}

void HashTable::Init(size_t table_size)
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

bool HashTable::Insert(const std::vector<Entry>& elements)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // 1. Allocate GPU memory for key-val-pairs to insert, padded to size of wavefront
    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(elements.size()), static_cast<uint32_t>(mpp::constants::WAVEFRONT_SIZE)));
    cl_mem key_val_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint64_t), NULL, NULL);

    // 2. Fill buffer

    status = clEnqueueWriteBuffer(mgr->command_queue, key_val_buffer, CL_TRUE, 0, elements.size() * sizeof(uint64_t), elements.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // If necessary write padding
    if (elements.size() < next_multiple)
    {
        uint32_t num_padded_elements = next_multiple - static_cast<uint32_t>(elements.size());
        std::vector<uint64_t> empty_elements(num_padded_elements);
        std::fill(empty_elements.begin(), empty_elements.end(), mpp::constants::EMPTY);
        size_t offset = elements.size() * sizeof(uint64_t);
        size_t num_bytes_written = num_padded_elements * sizeof(cl_int);
        status = clEnqueueWriteBuffer(mgr->command_queue, key_val_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // 3. Run kernel    
    const cl_kernel kernel_hashtable_insert = mgr->kernel_map[mpp::kernels::HASHTABLE_INSERT];
    //status = clSetKernelArg(kernel_prefix_scan, 0, sizeof(cl_mem), (void*)&a_buffer);
    //assert(status == mpp::ReturnCode::CODE_SUCCESS);
    //status = clSetKernelArg(kernel_prefix_scan, 1, sizeof(cl_mem), (void*)&b_buffer);
    //assert(status == mpp::ReturnCode::CODE_SUCCESS);
    //status = clSetKernelArg(kernel_prefix_scan, 2, sizeof(cl_mem), (void*)&c_buffer);
    //assert(status == mpp::ReturnCode::CODE_SUCCESS);

    //size_t global_work_size[1] = { static_cast<size_t>(next_multiple) };
    //size_t local_work_size[1] = { static_cast<size_t>(mpp::constants::MAX_THREADS_PER_CU) };
    //status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_hashtable_insert, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    //assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // TODO: Error checking (collisions) -> Reconstruction

    // 4. Cleanup -> Release buffers
    status = clReleaseMemObject(key_val_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    
    return status == mpp::ReturnCode::CODE_SUCCESS;
}

Entry Get(const std::vector<int32_t>& keys)
{
    // TODO

    return Entry();
}
