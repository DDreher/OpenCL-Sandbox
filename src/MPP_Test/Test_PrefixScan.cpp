#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Definitions.h"
#include "PrefixSum.h"
#include "OpenCLManager.h"

TEST_CASE("PrefixSum CPU", "[cpu]")
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_PREFIX_SUM, { mpp::kernels::PREFIX_SUM, mpp::kernels::PREFIX_CALC_E });

    std::vector<cl_int> test_elements;
    std::vector<cl_int> expected_output;

    SECTION("Size<100")
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

    SECTION("Size>256 && MultipleOf256")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 512; ++i)
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

    SECTION("val == 0")
    {
        // Fill input elements and expected output elements 
        cl_int tmp_num = 0;
        for (cl_int i = 0; i < 512; ++i)
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
        for (cl_int i = 0; i < 512; ++i)
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
