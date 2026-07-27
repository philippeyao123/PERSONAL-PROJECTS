#pragma once

#include <stdexcept>
#include <string>

namespace qf {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

class ValidationError : public Error {
 public:
  explicit ValidationError(const std::string& message) : Error(message) {}
};

class NumericalError : public Error {
 public:
  explicit NumericalError(const std::string& message) : Error(message) {}
};

}  // namespace qf
