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
        //uint32_t num_elements = 10;
        //HashTable hash_table(num_elements);

        //std::vector<uint32_t> keys = { 42 };
        //std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        //REQUIRE(retrieved_vals.size() == 1);
        //REQUIRE(retrieved_vals[0] == static_cast<uint32_t>(mpp::constants::EMPTY));
    }

    SECTION("Insert one element")
    {
        //uint32_t num_elements = 1;
        //HashTable hash_table(num_elements);

        //std::vector<Entry> elements;
        //Entry entry;
        //entry.key = 42;
        //entry.value = 123;
        //elements.push_back(entry);
        //hash_table.Insert(elements);
    }

    SECTION("Retrieve one element")
    {
        uint32_t num_elements = 1;
        HashTable hash_table(num_elements);

        std::vector<uint32_t> keys = { 1 };
        std::vector<uint32_t> values = { 2 };
        hash_table.Insert(keys, values);

        std::vector<uint32_t> retrieved_vals = hash_table.Get(keys);

        REQUIRE(retrieved_vals.size() == 1);
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
