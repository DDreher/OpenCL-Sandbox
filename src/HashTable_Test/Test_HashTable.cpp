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
    }

    SECTION("Insert one element")
    {
        size_t num_elements = 1;
    }

    SECTION("Retrieve one element")
    {
        size_t num_elements = 1;
    }

    SECTION("Insert thousand elements")
    {
        size_t num_elements = 1000;
    }

    SECTION("Retrieve thousand elements")
    {
        size_t num_elements = 1000;
    }
}
