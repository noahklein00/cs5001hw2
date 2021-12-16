#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

namespace ai
{

template<std::size_t numLayers>
weights<numLayers>::weights(const std::array<std::pair<std::size_t, std::size_t>, numLayers>& hold) {
    for(std::size_t i = 0; i < numLayers; ++i) {
        mesh[i] = std::vector<std::vector<double>>(hold[i].first);
        for(std::size_t j = 0; j < hold[i].first; ++j) {
            mesh[i][j] = std::vector<double>(hold[i].second);
            for(std::size_t k = 0; k < hold[i].second; ++k) {
                mesh[i][j][k] = static_cast<double>((rand() % 2000 - 1000) / 1000.0); // randomize weights
            }
        }
    }
}

template <std::size_t numLayers>
void weights<numLayers>::print() const noexcept {
    std::size_t layerCount = 0;
    std::size_t nodeCount = 0;
    for(const auto& layer : mesh) {
        std::cout << "Layer " << layerCount++ << "\n";
        for(const auto& node : layer) {
            std::cout << "\tNode " << nodeCount++ << "\n";
            std::cout << "\t\t";
            for(const auto& weight : node) {
                std::cout << weight << ", ";
            }
            std::cout << "\n";
        }
        nodeCount = 1;
    }
}

} // namespace ai

#endif // WEIGHTS_HPP