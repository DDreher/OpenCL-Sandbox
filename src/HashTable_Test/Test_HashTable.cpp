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
        HashTable hash_table(10);
    }

    SECTION("Insert one element")
    {
        size_t num_elements = 1;
        HashTable hash_table(num_elements);

        std::vector<Entry> elements;
        Entry entry;
        entry.key = 42;
        entry.value = 123;
        elements.push_back(entry);
        hash_table.Insert(elements);
    }

    SECTION("Retrieve one element")
    {
        size_t num_elements = 1;
        HashTable hash_table(num_elements);
    }

    SECTION("Insert thousand elements")
    {
        size_t num_elements = 1000;
        HashTable hash_table(num_elements);
    }

    SECTION("Retrieve thousand elements")
    {
        size_t num_elements = 1000;
        HashTable hash_table(num_elements);
    }

    SECTION("Insert a million elements")
    {
        size_t num_elements = 1'000'000;
        HashTable hash_table(num_elements);
    }

    SECTION("Retrieve a million elements")
    {
        size_t num_elements = 1'000'000;
        HashTable hash_table(num_elements);
    }
}
