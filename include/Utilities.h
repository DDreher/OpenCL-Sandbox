#pragma once
#include <utility>
#include <string>

enum ReturnCode;

class Utility
{
public:
    static std::pair<ReturnCode, std::string> ReadFile(const std::string& file_name);
};
