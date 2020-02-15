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
    Timer timer;
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(mpp::filenames::KERNELS_HASHTABLE, { mpp::kernels::HASHTABLE_INSERT, mpp::kernels::HASHTABLE_RETRIEVE});

    SECTION("Try to retrieve element from empty table")
    {
        std::vector<uint32_t> keys = { 42 };

        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()));
        REQUIRE(success == true);

        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals[0] == static_cast<uint32_t>(mpp::constants::EMPTY));
    }

    SECTION("Try to retrieve multiple elements from empty table")
    {
        std::vector<uint32_t> keys = { 4, 2, 3, 10, 11, 42 };

        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()));
        REQUIRE(success == true);

        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        REQUIRE(retrieved_vals.size() == keys.size());
        for(auto val : retrieved_vals)
        {
            REQUIRE(val == mpp::constants::EMPTY_32);
        }
    }

    SECTION("Insert one element")
    {
        std::cout << "----- Hashmap Insert - 1 element ----- " << std::endl;
        uint32_t num_elements = 1;
        std::vector<uint32_t> keys = { 42 };
        std::vector<uint32_t> values = { 123 };

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for(uint32_t i = 0; i<keys.size(); ++i)
        {
            cpu_hash.insert({keys[i], values[i]});
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == true);
    }

    SECTION("Retrieve one element")
    {
        std::cout << "----- Hashmap Retrieve - 1 element ----- " << std::endl;
        uint32_t num_elements = 1;
        std::vector<uint32_t> keys = { 42 };
        std::vector<uint32_t> values = { 123 };

        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t>::iterator it;
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            it = cpu_hash.find(keys[i]);
            if (it != cpu_hash.end())
            {
                uint32_t found_element = it->second;
            }
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;
        
        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        REQUIRE(success == true);

        timer.Reset();
        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals == values);
    }

    SECTION("Retrieve multiple elements")
    {
        std::cout << "----- Hashmap Insert - 8 elements ----- " << std::endl;
        uint32_t num_elements = 8;
        std::vector<uint32_t> keys = { 4, 2, 3, 10, 11, 42, 99, 102 };
        std::vector<uint32_t> values = { 123, 1, 8, 9, 20, 40, 4, 5 };
        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        REQUIRE(success == true);

        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t>::iterator it;
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            it = cpu_hash.find(keys[i]);
            if (it != cpu_hash.end())
            {
                uint32_t found_element = it->second;
            }
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals == values);
    }

    SECTION("Insert 1000 elements")
    {
        std::cout << "----- Hashmap Insert - 1000 elements ----- " << std::endl;
        uint32_t num_elements = 1000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == true);
    }

    SECTION("Retrieve 1000 elements")
    {
        std::cout << "----- Hashmap Retrieve - 1000 elements ----- " << std::endl;
        uint32_t num_elements = 1000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t>::iterator it;
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            it = cpu_hash.find(keys[i]);
            if (it != cpu_hash.end())
            {
                uint32_t found_element = it->second;
            }
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        REQUIRE(success == true);  // Increasing the number of possible evictions before abort leads to a successful hash table insertion

        timer.Reset();
        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals == values);
    }

    SECTION("Insert 1000, size_factor=4.0, max_reconstructions=10, max_iterations=7*log(N)")
    {
        std::cout << "----- Hashmap Insert - 1000 elements, size_factor=4.0, max_reconstructions=10, max_iterations=7*log(N) ----- " << std::endl;
        uint32_t num_elements = 1000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        hash_table.table_size_factor = 4.0f;
        hash_table.max_reconstructions = 10;
        hash_table.max_iterations = 7 * static_cast<uint32_t>(log(num_elements));
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == true);
    }

    SECTION("Insert a million elements")
    {
        std::cout << "----- Hashmap Insert - 1'000'000 elements (aborted insertion on GPU) ----- " << std::endl;
        uint32_t num_elements = 1'000'000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == false); // This should fail, the table can not be constructed without collisions
    }

    SECTION("Insert a million elements, max_iterations=7*log(N)")
    {
        std::cout << "----- Hashmap Insert - 1'000'000 elements, max_iterations=7*log(N) ----- " << std::endl;
        uint32_t num_elements = 1'000'000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        hash_table.max_iterations = 7 * static_cast<uint32_t>(log(num_elements));
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == true); // Increasing the number of possible evictions before abort leads to a successful hash table insertion
    }

    SECTION("Insert a million elements, size_factor=4.0, max_iterations=7*log(N)")
    {
        std::cout << "----- Hashmap Insert - 1'000'000 elements, size_factor=4.0, max_iterations=7*log(N) ----- " << std::endl;
        uint32_t num_elements = 1'000'000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        timer.Reset();
        HashTable hash_table;
        hash_table.table_size_factor = 4.0f;
        hash_table.max_iterations = 7 * static_cast<uint32_t>(log(num_elements));
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(success == true); // Increasing the number of possible evictions before abort leads to a successful hash table insertion
    }

    SECTION("Retrieve a million elements, max_iterations=7*log(N)")
    {
        std::cout << "----- Hashmap Retrieve - 1'000'000 elements, max_iterations=7*log(N) ----- " << std::endl;
        uint32_t num_elements = 1'000'000;
        std::vector<uint32_t> keys(num_elements);
        std::vector<uint32_t> values(num_elements);

        for (uint32_t i = 0; i < num_elements; ++i)
        {
            keys[i] = (i + 2);
            values[i] = (i + 3);
        }

        std::unordered_map<uint32_t, uint32_t> cpu_hash;
        cpu_hash.reserve(num_elements);
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            cpu_hash.insert({ keys[i], values[i] });
        }

        timer.Reset();
        std::unordered_map<uint32_t, uint32_t>::iterator it;
        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            it = cpu_hash.find(keys[i]);
            if(it != cpu_hash.end())
            {
                uint32_t found_element = it->second;
            }
        }
        std::cout << "Duration CPU: " << timer.GetElapsed() << " seconds" << std::endl;

        HashTable hash_table;
        hash_table.max_iterations = 7 * static_cast<uint32_t>(log(num_elements));
        bool success = hash_table.Init(static_cast<uint32_t>(keys.size()), keys, values);
        REQUIRE(success == true);  // Increasing the number of possible evictions before abort leads to a successful hash table insertion

        timer.Reset();
        std::vector<uint32_t> retrieved_vals = hash_table.Retrieve(keys);
        std::cout << "Duration GPU: " << timer.GetElapsed() << " seconds" << std::endl;
        REQUIRE(retrieved_vals.size() == keys.size());
        REQUIRE(retrieved_vals == values);
    }
}
