#include "PrefixSum.h"
#include "Definitions.h"
#include <assert.h>
#include <CL\cl.h>
#include "OpenCLManager.h"

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
    cl_int status = 0;
    cl_int num_elements = 256;

    // Create buffers
    cl_mem input_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY, num_elements * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr->command_queue, input_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), &elements, 0, NULL, NULL);
    assert(status != ReturnCode::ERROR);

    cl_mem output_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);

    // Run kernel 
    const cl_kernel kernel = mgr->kernel_map[constants::KERNEL_PREFIX_SUM];
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    assert(status != ReturnCode::ERROR);

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    assert(status != ReturnCode::ERROR);

    size_t global_work_size[1] = { num_elements };
    size_t local_work_size[1] = { num_elements };
    status = clEnqueueNDRangeKernel(mgr->command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status != ReturnCode::ERROR);

    // Get results
    std::vector<cl_int> output(elements.size(), 0);
    status = clEnqueueReadBuffer(mgr->command_queue, output_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), &output, 0, NULL, NULL);
    assert(status != ReturnCode::ERROR);

    // Clean up
    status = clReleaseMemObject(input_buffer);
    assert(status != ReturnCode::ERROR);
    status = clReleaseMemObject(output_buffer);
    assert(status != ReturnCode::ERROR);

    return output;
}
