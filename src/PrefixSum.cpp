#include "PrefixSum.h"
#include "Definitions.h"
#include "OpenCLManager.h"
#include "Utilities.h"
#include <assert.h>

std::vector<cl_int> PrefixSum::CalculateCPU(const std::vector<cl_int>& elements)
{
    uint32_t sum = 0;

    std::vector<cl_int> prefix_sum;
    prefix_sum.reserve(elements.size());

    for(cl_int element : elements)
    {
        prefix_sum.push_back(sum);
        sum += element;
    }

    return prefix_sum;
}

std::vector<cl_int> PrefixSum::CalculateGPU(const std::vector<cl_int>& elements)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    cl_int next_multiple = static_cast<cl_int>(
        Utility::GetNextMultipleOf(static_cast<uint32_t>(elements.size()), static_cast<uint32_t>(mpp::constants::MAX_THREADS_PER_CU)));
    
    // Allocate buffer A & B
    cl_mem input_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, next_multiple * sizeof(cl_int), NULL, NULL);           // Buffer A
    status = clEnqueueWriteBuffer(mgr->command_queue, input_buffer, CL_TRUE, 0, elements.size() * sizeof(cl_int), elements.data(), 0, NULL, NULL);
    cl_mem sub_prefix_sum_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, next_multiple * sizeof(cl_int), NULL, NULL);  // Buffer B

    // Fill padded memory with zeros
    if (elements.size() < next_multiple)
    {
        cl_int zeros[mpp::constants::MAX_THREADS_PER_CU] = { 0 };
        size_t offset = elements.size() * sizeof(cl_int);
        size_t num_bytes_written = (next_multiple - elements.size()) * sizeof(cl_int);
        status = clEnqueueWriteBuffer(mgr->command_queue, input_buffer, CL_TRUE, offset, num_bytes_written, &zeros, 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
    }

    // Call recursive function with A and B as input
    std::vector<cl_int> result = PrefixSum::CalculateGPU_Recursive(input_buffer, sub_prefix_sum_buffer, next_multiple);
    // Remove the padded values
    result.resize(elements.size());

    // Release buffers
    status = clReleaseMemObject(input_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clReleaseMemObject(sub_prefix_sum_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    return result;
}

std::vector<cl_int> PrefixSum::CalculateGPU_Recursive(cl_mem a_buffer, cl_mem b_buffer, cl_int num_elements)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // Allocate buffer C & D
    cl_int num_sub_arrays = num_elements / static_cast<cl_int>(mpp::constants::MAX_THREADS_PER_CU);
    cl_mem c_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_sub_arrays * sizeof(cl_int), NULL, NULL);
    cl_mem d_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_sub_arrays * sizeof(cl_int), NULL, NULL);

    // TODO: Pad to multiple of 256?

    // Prepare prefix scan kernel
    const cl_kernel kernel_prefix_scan = mgr->kernel_map[mpp::kernels::PREFIX_SUM];
    status = clSetKernelArg(kernel_prefix_scan, 0, sizeof(cl_mem), (void*)&a_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_prefix_scan, 1, sizeof(cl_mem), (void*)&b_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    status = clSetKernelArg(kernel_prefix_scan, 2, sizeof(cl_mem), (void*)&c_buffer);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // Run prefix scan kernel
    size_t global_work_size[1] = { static_cast<size_t>(num_elements) };
    size_t local_work_size[1] = { static_cast<size_t>(mpp::constants::MAX_THREADS_PER_CU) };    // Use a full wavefront/warp as local work size
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_prefix_scan, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    if(num_sub_arrays > mpp::constants::MAX_THREADS_PER_CU) // TODO: Checken
    {
        // the c / d buffer will be bigger than MAX_WORK_GROUP_SIZE
        // -> We have to do process these buffers recursively
        return CalculateGPU_Recursive(c_buffer, d_buffer, num_sub_arrays);
    }
    else
    {
        // c / d buffers are not bigger than a wavefront -> we can calculate final results
        cl_mem e_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        
        // TODO: Padding to next multiple of wavefront size?

        // Set kernel arguments.
        const cl_kernel kernel_calc_e = mgr->kernel_map[mpp::kernels::PREFIX_CALC_E];
        status = clSetKernelArg(kernel_calc_e, 0, sizeof(cl_mem), (void*)&b_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clSetKernelArg(kernel_calc_e, 1, sizeof(cl_mem), (void*)&d_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clSetKernelArg(kernel_calc_e, 2, sizeof(cl_mem), (void*)&e_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clSetKernelArg(kernel_calc_e, 3, sizeof(cl_uint), &num_elements);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // Run the kernel.
        size_t global_work_size[1] = { static_cast<size_t>(num_elements) };
        size_t local_work_size[1] = { static_cast<size_t>(mpp::constants::MAX_THREADS_PER_CU) };    // Use a full wavefront/warp as local work size
        status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_calc_e, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // Read results
        std::vector<cl_int> result(num_elements, 0);
        status = clEnqueueReadBuffer(mgr->command_queue, e_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), result.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // release buffers
        status = clReleaseMemObject(e_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(d_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(c_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        return result;
    }
}
