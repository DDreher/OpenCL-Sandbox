#pragma once
#include <vector>
#include <cstdint>
#include <CL\cl.h>
#include "Base/Definitions.h"

union Entry
{
    struct
    {
        uint32_t key : 32;
        uint32_t value : 32;
    };

    uint64_t data;
};

class HashTable
{
public:
    HashTable(size_t size);
    ~HashTable();

    void Init(size_t table_size);
    bool Insert(const std::vector<Entry>& elements);
    Entry Get(const std::vector<uint32_t>& keys);

private:
    size_t size_ = 0;
    cl_mem table_buffer_ = 0;
    const uint32_t THREAD_BLOCK_SIZE = 64;

    // parameters
    size_t PARAM_IDX_HASHFUNC_A_0 = 0;
    size_t PARAM_IDX_HASHFUNC_B_0 = 1;
    size_t PARAM_IDX_HASHFUNC_A_1 = 2;
    size_t PARAM_IDX_HASHFUNC_B_1 = 3;
    size_t PARAM_IDX_HASHFUNC_A_2 = 4;
    size_t PARAM_IDX_HASHFUNC_B_2 = 5;
    size_t PARAM_IDX_HASHFUNC_A_3 = 6;
    size_t PARAM_IDX_HASHFUNC_B_3 = 7;
    size_t PARAM_IDX_MAX_ITERATIONS = 8;
    size_t PARAM_IDX_TABLESIZE = 9;
    uint32_t params[10] = { 0 };
};
