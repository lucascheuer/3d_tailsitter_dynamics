#pragma once
#include <eigen3/Eigen/Eigen>

#include "toml.hpp"

namespace TomlParseHelpers
{
double ParseDouble(toml::table& tbl, std::string key);

Eigen::Vector3d ParseDoubleVector3d(toml::table& tbl, std::string key);
};  // namespace TomlParseHelpers