#include "Utilities.h"
#include "Definitions.h"
#include <string>
#include <fstream>
#include "assert.h"

std::pair<ReturnCode, std::string> Utility::ReadFile(const std::string& file_name)
{
    std::string out_string = "";
    ReturnCode return_code = ReturnCode::ERROR;

    std::fstream file_stream(file_name, (std::fstream::in | std::fstream::binary));
    if (file_stream.is_open())
    {
        size_t file_size;
        file_stream.seekg(0, std::fstream::end);
        file_size = static_cast<size_t>(file_stream.tellg());
        file_stream.seekg(0, std::fstream::beg);

        char* str = new char[file_size + 1];    // +1 for delimiter
        file_stream.read(str, file_size);
        file_stream.close();
        str[file_size] = '\0';
        out_string = str;
        delete[] str;

        return_code = ReturnCode::SUCCESS;
    }

    return { return_code, out_string };
}

uint32_t Utility::GetNextMultipleOf(uint32_t num_to_round, uint32_t num_multiple)
{
    assert(num_to_round!= 0);
    return ((num_to_round + num_multiple - 1) / num_multiple) * num_multiple;
}
