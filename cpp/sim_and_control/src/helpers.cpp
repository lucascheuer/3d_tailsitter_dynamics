#include "helpers.hpp"

#include <iostream>

double TomlParseHelpers::ParseDouble(toml::table& tbl, std::string key)
{
    std::optional<double> temp = tbl[key].value<double>();
    if (temp)
    {
        return *temp;
    } else
    {
        return std::nanf("");
    }
}

Eigen::Vector3d TomlParseHelpers::ParseDoubleVector3d(toml::table& tbl, std::string key)
{
    if (auto temp_array = tbl[key].as_array())
    {
        std::vector<double> temp_vector;
        for (auto& elem : *temp_array)
        {
            if (auto double_val = elem.value<double>())
            {
                temp_vector.push_back(*double_val);
            } else
            {
                std::cerr << "Error: Array element is not a double." << std::endl;
            }
        }
        return Eigen::Map<Eigen::Vector3d>(temp_vector.data());
    } else
    {
        std::cout << "failed to parse array" << std::endl;
        return Eigen::Vector3d::Zero();
    }
}
