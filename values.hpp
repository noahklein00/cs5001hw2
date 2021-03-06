#ifndef VALUES_HPP
#define VALUES_HPP

namespace ai
{

template <std::size_t numLayers>
void ai::values<numLayers>::setColumn(const std::size_t index, const std::vector<double>& col) {
    outputs[index] = col;
    return;
}

template <std::size_t numLayers>
void ai::values<numLayers>::print() const noexcept {
    for(const auto& layer : outputs) {
        for(const auto& output : layer) {
            std::cout << output << ", ";
        }
        std::cout << "\n";
    }
}

} // namespace ai

#endif