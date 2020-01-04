#pragma once
#include <vector>
#include <CL\cl.h>

struct Entry
{
    int32_t key;
    int32_t value;
};

class HashTable
{
public:
    HashTable(uint32_t size);
    ~HashTable();

    bool Insert(const std::vector<Entry>& elements);
    Entry Get(const std::vector<Entry>& elements);

    uint32_t size_ = 0;
};