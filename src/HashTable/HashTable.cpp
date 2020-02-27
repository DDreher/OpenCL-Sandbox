#include "HashTable/HashTable.h"
#include <Base\OpenCLManager.h>
#include "assert.h"
#include "Base\Definitions.h"
#include "Base\Utilities.h"

HashTable::HashTable()
{
    // Init random seed
    srand(random_seed_);
}

HashTable::~HashTable()
{
    // Release buffers
    cl_int status = 0;
    if(table_buffer_ != 0)
    {
        status = clReleaseMemObject(table_buffer_);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    if (params_buffer_ != 0)
    {
        status = clReleaseMemObject(params_buffer_);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }
}

bool HashTable::Init(uint32_t table_size)
{
    size_ = static_cast<uint32_t>(ceil(table_size * table_size_factor));
    GenerateParams();

    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // 1. Allocate enough memory on GPU to fit hash table
    if(table_buffer_ == 0)
    {
        table_buffer_ = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, size_ * sizeof(uint64_t), NULL, NULL);
    }

    // 2. Initialize all the memory with empty elements
    std::vector<uint64_t> empty_elements(size_, mpp::constants::EMPTY);
    status = clEnqueueWriteBuffer(mgr->command_queue, table_buffer_, CL_TRUE, 0, size_ * sizeof(uint64_t), empty_elements.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    return status == mpp::ReturnCode::CODE_SUCCESS;
}

bool HashTable::Init(uint32_t table_size, const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values)
{
    bool success = true;
    for(current_iteration_ = 0; current_iteration_ < max_reconstructions; ++current_iteration_)
    {
        Init(table_size);

        if(keys.size() > 0)
        {
            success = Insert(keys, values);

            if(success)
            {
                break;
            }
        }
    }

    return success;
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
    cl_mem values_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, next_multiple * sizeof(uint32_t), NULL, NULL);

    // 2. Fill buffers
    status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, 0, keys.size() * sizeof(uint32_t), keys.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clEnqueueWriteBuffer(mgr->command_queue, values_buffer, CL_TRUE, 0, values.size() * sizeof(uint32_t), values.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // If necessary write padding
    if (keys.size() < next_multiple)
    {
        uint32_t num_padded_elements = next_multiple - static_cast<uint32_t>(keys.size());
        std::vector<uint32_t> empty_elements(num_padded_elements, mpp::constants::EMPTY_32);
        size_t offset = keys.size() * sizeof(uint32_t);
        size_t num_bytes_written = num_padded_elements * sizeof(uint32_t);
        status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clEnqueueWriteBuffer(mgr->command_queue, values_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // 3. Construct status buffer
    cl_mem status_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, NULL);
    int intital_status = 0;
    status = clEnqueueWriteBuffer(mgr->command_queue, status_buffer, CL_TRUE, 0, sizeof(uint32_t), &intital_status, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 4. Run kernel    
    const cl_kernel kernel_hashtable_insert = mgr->kernel_map[mpp::kernels::HASHTABLE_INSERT];
    // args: __global uint32_t* keys, __global uint32_t* values, __global uint64_t* table, __constant uint32_t* params, __global uint32_t* out_status
    status = clSetKernelArg(kernel_hashtable_insert, 0, sizeof(cl_mem), (void*)&keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 1, sizeof(cl_mem), (void*)&values_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 2, sizeof(cl_mem), (void*)&table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 3, sizeof(cl_mem), (void*)&params_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_insert, 4, sizeof(cl_mem), (void*)&status_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    size_t global_work_size[1] = { static_cast<size_t>(next_multiple) };
    size_t local_work_size[1] = { std::min(static_cast<size_t>(THREAD_BLOCK_SIZE), static_cast<size_t>(next_multiple)) };
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_hashtable_insert, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 5. Error checking - Check if max iterations have been exceeded
    // In theory we could check which elements failed to be inserted, then take the original state of the hashtable, generate new hash parameters and then reconstruct the hashtable
    // from scratch. There probably are elements left which could not be inserted due to collision though...
    // For now it's assumed that insert is only called once.

    uint32_t kernel_status = mpp::ReturnCode::CODE_SUCCESS;
    status = clEnqueueReadBuffer(mgr->command_queue, status_buffer, CL_TRUE, 0, sizeof(uint32_t), &kernel_status, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // DEBUG - Check which/how many elements failed to be inserted
    //if(kernel_status != mpp::ReturnCode::CODE_SUCCESS)
    //{
    //    std::vector<uint32_t> status_per_element(next_multiple);
    //    status = clEnqueueReadBuffer(mgr->command_queue, keys_buffer, CL_TRUE, 0, next_multiple * sizeof(uint32_t), status_per_element.data(), 0, NULL, NULL);
    //    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    //    // Assume that 0 and 1 are never used as keys for debug purposes
    //    uint32_t num_unresolved_collisions = static_cast<uint32_t>(std::count(status_per_element.begin(), status_per_element.end(), mpp::ReturnCode::CODE_ERROR));
    //    std::cout << "Host Table construction iteration: " << current_iteration_ << " Num unresolved collisions: " << num_unresolved_collisions << std::endl;
    //}

    // 6. Cleanup -> Release buffers
    status = clReleaseMemObject(keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(values_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(status_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    
    return kernel_status == mpp::ReturnCode::CODE_SUCCESS;
}

std::vector<uint32_t> HashTable::Retrieve(const std::vector<uint32_t>& keys)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // 1. Allocate GPU memory
    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(keys.size()), static_cast<uint32_t>(mpp::constants::WAVEFRONT_SIZE)));
    cl_mem keys_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, next_multiple * sizeof(uint32_t), NULL, NULL);
    cl_mem vals_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(uint32_t), NULL, NULL);

    // 2. Fill buffer
    status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, 0, keys.size() * sizeof(uint32_t), keys.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // If necessary write padding
    if (keys.size() < next_multiple)
    {
        uint32_t num_padded_elements = next_multiple - static_cast<uint32_t>(keys.size());
        std::vector<uint32_t> empty_elements(num_padded_elements, mpp::constants::EMPTY_32);
        size_t offset = keys.size() * sizeof(uint32_t);
        size_t num_bytes_written = num_padded_elements * sizeof(uint32_t);
        status = clEnqueueWriteBuffer(mgr->command_queue, keys_buffer, CL_TRUE, offset, num_bytes_written, empty_elements.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // 3. Construct parameters buffer
    cl_mem params_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, params_.size() * sizeof(uint32_t), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, params_buffer, CL_TRUE, 0, params_.size() * sizeof(uint32_t), params_.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 4. invoke retrieve kernel
    const cl_kernel kernel_hashtable_retrieve = mgr->kernel_map[mpp::kernels::HASHTABLE_RETRIEVE];
    // params: __global int32_t* keys, __global int64_t* table, __constant uint32_t* params
    status = clSetKernelArg(kernel_hashtable_retrieve, 0, sizeof(cl_mem), (void*)&keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_retrieve, 1, sizeof(cl_mem), (void*)&vals_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_retrieve, 2, sizeof(cl_mem), (void*)&table_buffer_);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_hashtable_retrieve, 3, sizeof(cl_mem), (void*)&params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    size_t global_work_size[1] = { static_cast<size_t>(next_multiple) };
    size_t local_work_size[1] = { std::min(static_cast<size_t>(THREAD_BLOCK_SIZE), static_cast<size_t>(next_multiple)) };

    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_hashtable_retrieve, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 5. Read back results
    std::vector<uint32_t> retrieved_entries(next_multiple);
    status = clEnqueueReadBuffer(mgr->command_queue, vals_buffer, CL_TRUE, 0, next_multiple * sizeof(uint32_t), retrieved_entries.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // 6. Release buffers
    status = clReleaseMemObject(keys_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(vals_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(params_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    retrieved_entries.resize(keys.size());
    return retrieved_entries;
}

void HashTable::GenerateParams()
{
    params_[PARAM_IDX_HASHFUNC_A_0] = rand();
    params_[PARAM_IDX_HASHFUNC_B_0] = rand();
    params_[PARAM_IDX_HASHFUNC_A_1] = rand();
    params_[PARAM_IDX_HASHFUNC_B_1] = rand();
    params_[PARAM_IDX_HASHFUNC_A_2] = rand();
    params_[PARAM_IDX_HASHFUNC_B_2] = rand();
    params_[PARAM_IDX_HASHFUNC_A_3] = rand();
    params_[PARAM_IDX_HASHFUNC_B_3] = rand();
    params_[PARAM_IDX_MAX_ITERATIONS] = max_iterations;
    params_[PARAM_IDX_TABLESIZE] = size_;

    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    if (params_buffer_ != 0)
    {
        status = clReleaseMemObject(params_buffer_);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    params_buffer_ = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, params_.size() * sizeof(uint32_t), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, params_buffer_, CL_TRUE, 0, params_.size() * sizeof(uint32_t), params_.data(), 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
}
