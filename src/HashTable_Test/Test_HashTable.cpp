#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Base/Definitions.h"
#include "Base/OpenCLManager.h"
#include "Base/Utilities.h"
#include "HashTable/HashTable.h"

#include <stdio.h>
#include <iostream>

TEST_CASE("HashTable", "[gpu]")
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_HASHTABLE, { mpp::kernels::HASHTABLE_INSERT, mpp::kernels::HASHTABLE_RETRIEVE});

    SECTION("Try to retrieve element from empty table")
    {
        uint32_t num_elements = 10;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 42 };
        std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals[0] == static_cast<uint32_t>(mpp::constants::EMPTY));
    }

    SECTION("Try to retrieve multiple elements from empty table")
    {
        uint32_t num_elements = 10;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 1,2,3,10,11,42 };
        std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        REQUIRE(retrieved_vals.size() == keys.size());
        for(auto val : retrieved_vals)
        {
            REQUIRE(val == mpp::constants::EMPTY_32);
        }
    }

    SECTION("Insert one element")
    {
        uint32_t num_elements = 1;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 1 };
        std::vector<uint32_t> values = { 123 };
        hash_table.Insert(keys, values);
    }

    SECTION("Retrieve one element")
    {
        uint32_t num_elements = 1;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 42 };
        std::vector<uint32_t> values = { 123 };
        hash_table.Insert(keys, values);

        std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        REQUIRE(retrieved_vals.size() == 1);
        REQUIRE(retrieved_vals == values);
    }

    SECTION("Retrieve multiple elements")
    {
        uint32_t num_elements = 1;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 1, 2, 3, 10, 11, 42 };
        std::vector<uint32_t> values = { 123, 1, 8, 9, 20, 40 };
        hash_table.Insert(keys, values);

        std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals == values);
    }

    SECTION("Insert thousand elements")
    {
        uint32_t num_elements = 1000;
        HashTable hash_table(num_elements);
    }

    SECTION("Retrieve thousand elements")
    {
        uint32_t num_elements = 1000;
        HashTable hash_table(num_elements);
    }

    SECTION("Insert a million elements")
    {
        uint32_t num_elements = 1'000'000;
        HashTable hash_table(num_elements);
    }

    SECTION("Retrieve a million elements")
    {
        uint32_t num_elements = 1'000'000;
        HashTable hash_table(num_elements);
    }
}
