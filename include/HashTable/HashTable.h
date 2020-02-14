#pragma once
#include <vector>
#include <cstdint>
#include <CL\cl.h>
#include "Base/Definitions.h"

class HashTable
{
public:
    HashTable();
    ~HashTable();

    bool Init(uint32_t table_size);
    bool Init(uint32_t table_size, const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values);
    bool Insert(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values);
    std::vector<uint32_t> Get(const std::vector<uint32_t>& keys);

    uint32_t max_iterations = 8;
    uint32_t max_reconstructions = 3;
    float table_size_factor = 1.25f;

private:
    void GenerateParams();

    cl_mem table_buffer_ = 0;
    const uint32_t THREAD_BLOCK_SIZE = 64;

    // parameters
    uint32_t size_ = 0;

    const uint32_t NUM_PARAMS = 10;
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
    std::vector<uint32_t> params_;
};
