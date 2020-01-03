#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Definitions.h"
#include "PrefixSum.h"
#include "OpenCLManager.h"
#include "Utilities.h"

#include <stdio.h>
#include <iostream>

TEST_CASE("PrefixSum CPU", "[cpu]")
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_PREFIX_SUM, { mpp::kernels::PREFIX_SUM, mpp::kernels::PREFIX_CALC_E });

    std::vector<cl_int> test_elements;
    std::vector<cl_int> expected_output;

    SECTION("Size==256")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 256; ++i)
        {
            cl_int val = 1;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("Size<256")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 100; ++i)
        {
            cl_int val = 1;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("Size>256 && MultipleOf256")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 512; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    }

    SECTION("Size>256")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 1000; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("val == 0")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 42; ++i)
        {
            cl_int val = 0;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("val < 0")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 1337; ++i)
        {
            cl_int val = -i;
            test_elements.push_back(val);
            expected_output.push_back(tmp_num);
            tmp_num += val;
        }

        // Test the output
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
        REQUIRE(test_prefix_sum == expected_output);
    };
};

TEST_CASE("PrefixSum Kernel Calculate e_buffer", "[kernel e_buffer]")
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_PREFIX_SUM, { mpp::kernels::PREFIX_SUM, mpp::kernels::PREFIX_CALC_E });

    std::vector<cl_int> vec_b_buffer;
    std::vector<cl_int> vec_d_buffer;
    std::vector<cl_int> expected_output;
    size_t num_elements = 0;
    cl_int status = 0;

    SECTION("global_work_size=8, local_work_size=4")
    {
        // Fill input & expected result
        vec_b_buffer = { 0,3,5,6,0,1,5,8 };
        vec_d_buffer = { 0,8 };
        expected_output = { 0,3,5,6,8,9,13,16 };

        num_elements = vec_b_buffer.size();

        // Allocate buffers
        cl_mem b_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        status = clEnqueueWriteBuffer(mgr->command_queue, b_buffer, CL_TRUE, 0, vec_b_buffer.size() * sizeof(cl_int), vec_b_buffer.data(), 0, NULL, NULL);
        cl_mem d_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        status = clEnqueueWriteBuffer(mgr->command_queue, d_buffer, CL_TRUE, 0, vec_d_buffer.size() * sizeof(cl_int), vec_d_buffer.data(), 0, NULL, NULL);
        cl_mem e_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        
        // Set kernel args
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
        size_t local_work_size[1] = { static_cast<size_t>(4) };
        status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_calc_e, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // Read results
        std::vector<cl_int> result(num_elements, 0);
        status = clEnqueueReadBuffer(mgr->command_queue, e_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), result.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // release buffers
        status = clReleaseMemObject(b_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(d_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(e_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        REQUIRE(result == expected_output);
    };

    SECTION("global_work_size=512, local_work_size=256")
    {
        size_t total_size = 512;
        size_t group_size = 256;
        
        // Fill input & expected result
        std::vector<cl_int> test_elements(total_size, 1);
        expected_output = PrefixSum::CalculateCPU(test_elements);

        cl_int tmp = 0;
        for (cl_int i = 0; i < total_size; ++i)
        {
            if(tmp % group_size == 0)
            {
                tmp = 0;
            }

            vec_b_buffer.push_back(tmp);
            tmp += test_elements[i];
        }

        for(size_t i = group_size-1; i < total_size; i += group_size)
        {
            vec_d_buffer.push_back(test_elements[i] + vec_b_buffer[i]);
        }
        vec_d_buffer = PrefixSum::CalculateCPU(vec_d_buffer);

        num_elements = vec_b_buffer.size();

        // Allocate buffers
        cl_mem b_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        status = clEnqueueWriteBuffer(mgr->command_queue, b_buffer, CL_TRUE, 0, vec_b_buffer.size() * sizeof(cl_int), vec_b_buffer.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        cl_mem d_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);
        status = clEnqueueWriteBuffer(mgr->command_queue, d_buffer, CL_TRUE, 0, vec_d_buffer.size() * sizeof(cl_int), vec_d_buffer.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        cl_mem e_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), NULL, NULL);

        // Set kernel args
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
        size_t local_work_size[1] = { static_cast<size_t>(mpp::constants::MAX_THREADS_PER_CU) };
        status = clEnqueueNDRangeKernel(mgr->command_queue, kernel_calc_e, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // Read results
        std::vector<cl_int> result(num_elements, 0);
        status = clEnqueueReadBuffer(mgr->command_queue, e_buffer, CL_TRUE, 0, num_elements * sizeof(cl_int), result.data(), 0, NULL, NULL);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        // release buffers
        status = clReleaseMemObject(b_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(d_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        status = clReleaseMemObject(e_buffer);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        REQUIRE(result == expected_output);
    };
}

TEST_CASE("PrefixSum GPU", "[gpu]")
{
    Timer timer;

    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_PREFIX_SUM, { mpp::kernels::PREFIX_SUM, mpp::kernels::PREFIX_CALC_E });

    std::vector<cl_int> test_elements;
    std::vector<cl_int> expected_output;

    SECTION("Size<256")
    {
        std::cout << "--------------- PrefixScan - 100 elements --------------- " << std::endl;

        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 100; ++i)
        {
            cl_int val = 1;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("Size==256")
    {
        std::cout << "--------------- PrefixScan - 256 elements --------------- " << std::endl;

        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 256; ++i)
        {
            cl_int val = 1;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("Size>256 && MultipleOf256")
    {
        std::cout << "--------------- PrefixScan - 512 elements --------------- " << std::endl;

        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 512; ++i)
        {
            cl_int val = 1;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("Size>256 && NoMultipleOf256")
    {
        std::cout << "--------------- PrefixScan - 1000 elements --------------- " << std::endl;

        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 1000; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("val == 0")
    {
        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 512; ++i)
        {
            cl_int val = 0;
            test_elements.push_back(val);
        }
        expected_output = PrefixSum::CalculateCPU(test_elements);

        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("val < 0")
    {
        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < 512; ++i)
        {
            cl_int val = -i;
            test_elements.push_back(val);
        }
        expected_output = PrefixSum::CalculateCPU(test_elements);

        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("size == 1_000_000")
    {
        std::cout << "--------------- PrefixScan - 1'000'000 elements --------------- " << std::endl;
        size_t size = 1'000'000;
        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < size; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("size == 10'000'000")
    {
        std::cout << "--------------- PrefixScan - 10'000'000 elements --------------- " << std::endl;
        size_t size = 10'000'000;
        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < size; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
        }
        
        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
    };

    SECTION("size == 100'000'000")
    {
        std::cout << "--------------- PrefixScan - 10'000'000 elements --------------- " << std::endl;
        size_t size = 100'000'000;
        // Fill input elements and expected output elements 
        for (cl_int i = 0; i < size; ++i)
        {
            cl_int val = i;
            test_elements.push_back(val);
        }

        timer.Reset();
        expected_output = PrefixSum::CalculateCPU(test_elements);
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;

        // Test the output
        REQUIRE(test_prefix_sum == expected_output);
        /*std::cout << "Press enter to close..." << std::endl;
        std::cin.get();*/
    };
};
