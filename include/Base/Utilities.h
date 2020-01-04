#pragma once
#include <utility>
#include <string>
#include <iostream>
#include <chrono>

#include "Base/Definitions.h"

class Utility
{
public:
    static std::pair<mpp::ReturnCode, std::string> ReadFile(const std::string& file_name);
    static uint32_t GetNextMultipleOf(uint32_t num_to_round, uint32_t num_multiple);
};

class Timer
{
public:
    Timer() : timestamp_begin_(clock_::now()) {}

    void Reset()
    {
        timestamp_begin_ = clock_::now();
    }

    double GetElapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - timestamp_begin_).count();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> timestamp_begin_;
};
