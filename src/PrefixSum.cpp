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

    uint32_t next_multiple = Utility::GetNextMultipleOf(elements.size(), constants::MAX_WORK_GROUP_SIZE);
    
    // Allocate buffer A & B
    cl_mem input_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, next_multiple * sizeof(cl_int), NULL, NULL);           // Buffer A
    cl_mem sub_prefix_sum_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, next_multiple * sizeof(cl_int), NULL, NULL);  // Buffer B

    // Fill padded memory with zeros
    uint32_t input_size = elements.size();
    if (input_size < next_multiple)
    {
        cl_int zeros[constants::MAX_WORK_GROUP_SIZE] = { 0 };
        size_t offset = input_size * sizeof(cl_int);
        size_t num_bytes_written = (next_multiple - input_size) * sizeof(cl_int);
        status = clEnqueueWriteBuffer(mgr->command_queue, input_buffer, CL_TRUE, offset, num_bytes_written, &zeros, 0, NULL, NULL);
        assert(status == ReturnCode::SUCCESS);
    }

    // Call recursive function with A and B as input
    std::vector<cl_int> result = PrefixSum::CalculateGPU_Recursive(input_buffer, sub_prefix_sum_buffer, next_multiple);
    // Remove the padded values
    result.resize(elements.size());

    // Release buffers
    status = clReleaseMemObject(input_buffer);
    assert(status == ReturnCode::SUCCESS);
    status = clReleaseMemObject(sub_prefix_sum_buffer);
    assert(status == ReturnCode::SUCCESS);

    return result;
}

std::vector<cl_int> PrefixSum::CalculateGPU_Recursive(cl_mem a_buffer, cl_mem b_buffer, size_t num_elements)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    assert(mgr != nullptr);
    cl_int status = 0;

    // Allocate buffer C & D
    uint32_t num_sub_arrays = num_elements / constants::MAX_WORK_GROUP_SIZE;
    cl_mem c_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_sub_arrays * sizeof(cl_int), NULL, NULL);
    cl_mem d_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_sub_arrays * sizeof(cl_int), NULL, NULL);

    // TODO: Pad to multiple of 256?

    // Prepare prefix scan kernel
    const cl_kernel kernel_prefix_scan = mgr->kernel_map[kernels::PREFIX_SUM];
    status = clSetKernelArg(kernel_prefix_scan, 0, sizeof(cl_mem), (void*)&a_buffer);
    assert(status == ReturnCode::SUCCESS);
    status = clSetKernelArg(kernel_prefix_scan, 1, sizeof(cl_mem), (void*)&b_buffer);
    assert(status == ReturnCode::SUCCESS);
    status = clSetKernelArg(kernel_prefix_scan, 2, sizeof(cl_mem), (void*)&c_buffer);
    assert(status == ReturnCode::SUCCESS);

    // Run prefix scan kernel
    size_t global_work_size[1] = { num_sub_arrays };
    size_t local_work_size[1] = { static_cast<size_t>(constants::MAX_WORK_GROUP_SIZE) };    // Use a full wavefront/warp as local work size
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_prefix_scan, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    if(num_sub_arrays > constants::MAX_WORK_GROUP_SIZE)
    {
        // the c / d buffer will be bigger than MAX_WORK_GROUP_SIZE
        // -> We have to do process these buffers recursively
        return CalculateGPU_Recursive(c_buffer, d_buffer, num_sub_arrays);
    }
    else
    {
        // c / d buffers are not bigger than a wavefront -> we can calculate final results
        cl_mem e_buffer = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, num_elements * sizeof(cl_int), NULL, NULL);
        
        // TODO: Padding to next multiple of wavefront size?

        // Set kernel arguments.
        const cl_kernel kernel_calc_e = mgr->kernel_map[kernels::PREFIX_CALC_E];
        status = clSetKernelArg(kernel_calc_e, 0, sizeof(cl_mem), (void*)&d_buffer);
        assert(status == ReturnCode::SUCCESS);
        status = clSetKernelArg(kernel_calc_e, 1, sizeof(cl_mem), (void*)&e_buffer);
        assert(status == ReturnCode::SUCCESS);

        // Run the kernel.
        size_t global_work_size[1] = { num_elements };
        size_t local_work_size[1] = { static_cast<size_t>(constants::MAX_WORK_GROUP_SIZE) };    // Use a full wavefront/warp as local work size
        status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_calc_e, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        assert(status == ReturnCode::SUCCESS);

        // Read results
        std::vector<cl_int> result(num_elements, 0);
        status = clEnqueueReadBuffer(mgr->command_queue, e_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), result.data(), 0, NULL, NULL);
        assert(status == ReturnCode::SUCCESS);

        // release buffers
        status = clReleaseMemObject(e_buffer);
        assert(status == ReturnCode::SUCCESS);
        status = clReleaseMemObject(d_buffer);
        assert(status == ReturnCode::SUCCESS);
        status = clReleaseMemObject(c_buffer);
        assert(status == ReturnCode::SUCCESS);

        return result;
    }
}

// size of arrays must be exactly 256
int PrefixSum::praefixsumme(cl_int* input, cl_int* output, int size)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    cl_int status;

    int clsize = 256;

    // create OpenClinput buffer
    cl_mem inputBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, inputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), input, 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    // create OpenCl buffer for output
    cl_mem outputBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

    // Set kernel arguments.
    const cl_kernel kernel = mgr->kernel_map[kernels::PREFIX_SUM];
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputBuffer);
    assert(status == ReturnCode::SUCCESS);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outputBuffer);
    assert(status == ReturnCode::SUCCESS);

    // Run the kernel.
    size_t global_work_size[1] = { static_cast<size_t>(clsize) };
    size_t local_work_size[1] = { static_cast<size_t>(clsize) };
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    // get resulting array
    status = clEnqueueReadBuffer(mgr->command_queue, outputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output, 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    // release buffers
    status = clReleaseMemObject(inputBuffer);
    assert(status == ReturnCode::SUCCESS);

    status = clReleaseMemObject(outputBuffer);
    assert(status == ReturnCode::SUCCESS);
    
    return ReturnCode::SUCCESS;
}
