#pragma once
#include <utility>
#include <string>

enum ReturnCode;

class Utility
{
    static std::pair<ReturnCode, std::string> ReadFile(const std::string& file_name);
};
