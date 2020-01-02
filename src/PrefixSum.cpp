#include "PrefixSum.h"
#include "Definitions.h"
#include "OpenCLManager.h"
#include "Utilities.h"
#include <assert.h>
#include <CL\cl.h>

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

std::vector<cl_int> PrefixSum::CalculateGPU_256(const std::vector<cl_int>& elements)
{
    // TODO: base algorithm
    // TODO: pad buffer to multiple of 256
}

std::vector<cl_int> PrefixSum::CalculateGPU(const std::vector<cl_int>& elements)
{
    if(elements.size() <= constants::MAX_FIELD_SIZE)
    {
        // We can calculate the prefix sum with our base algorithm
        return PrefixSum::CalculateGPU_256(elements);
    }
    else
    {
        // split input array into multiples of MAX_FIELD_SIZE

        // For each sub array calculate the prefix sum with the base algorithm

        std::vector<std::vector<cl_int>> sub_arrays;
        for(auto arr : sub_arrays)
        {
            std::vector<cl_int> sub_array_result = CalculateGPU_256(arr);
        }


    }


    uint32_t next_multiple = Utility::GetNextMultipleOf(257, 256);

    OpenCLManager* mgr = OpenCLManager::GetInstance();
    cl_int status = 0;
    cl_int num_elements = 256;
    cl_int next_power_of_two = 256;

    // Create buffers
    cl_mem input_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, num_elements * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, input_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), elements.data(), 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    cl_mem output_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);

    // Run kernel 
    const cl_kernel kernel = mgr->kernel_map[kernels::PREFIX_SUM];
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &input_buffer);
    assert(status == ReturnCode::SUCCESS);

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &output_buffer);
    assert(status == ReturnCode::SUCCESS);

    size_t global_work_size[1] = { static_cast<size_t>(num_elements) };
    size_t local_work_size[1] = { static_cast<size_t>(num_elements) };
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    // Get results
    std::vector<cl_int> output(num_elements, 0);
    status = clEnqueueReadBuffer(mgr->command_queue, output_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), output.data(), 0, NULL, NULL);
    assert(status == ReturnCode::SUCCESS);

    // Clean up
    status = clReleaseMemObject(input_buffer);
    assert(status == ReturnCode::SUCCESS);
    status = clReleaseMemObject(output_buffer);
    assert(status == ReturnCode::SUCCESS);

    return output;
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
