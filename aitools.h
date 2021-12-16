#ifndef AITOOLS_H
#define AITOOLS_H

#include <array>
#include <vector>
#include <iostream>

namespace ai
{

using VTD = std::vector<std::vector<double>>;

template <std::size_t numLayers>
class weights {
    public:
        weights(const std::array<ai::VTD, numLayers>& hold) : mesh{hold} {}
        weights(const std::array<std::pair<std::size_t, std::size_t>, numLayers>& hold);
        void print() const noexcept;
        ai::VTD& operator[](const std::size_t index) { return mesh[index]; }
        const ai::VTD& operator[](const std::size_t index) const { return mesh[index]; }
    private:
        std::array<ai::VTD, numLayers> mesh;
};

template <std::size_t numLayers>
class values {
    public:
        void setColumn(const std::size_t index, const std::vector<double>& col);
        void print() const noexcept;
        std::vector<double>& operator[](const std::size_t index) { return outputs[index]; }
        const std::vector<double>& operator[](const std::size_t index) const { return outputs[index]; }
        std::size_t size() const noexcept { return outputs.size(); }
    private:
        std::array<std::vector<double>, numLayers> outputs;
};


} // namespace ai

#include "weights.hpp"
#include "values.hpp"

#endif // AITOOLS_H