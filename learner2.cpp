#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <cmath>
#include "aitools.h"

ai::VTD dataReader(std::istream& in) {
    constexpr char DELIMITER = '\t';
    std::string curLine;
    ai::VTD data;
    std::size_t index = 0;

    while(std::getline(in, curLine) && curLine.size() > 0) {
        std::stringstream unsplitLine(curLine);
        std::string catchData;
        data.push_back({});
        // data[index].push_back(1); // x0 is always 1

        while(std::getline(unsplitLine, catchData, DELIMITER)) {
            data[index].push_back(std::stod(catchData));
        }
        ++index;
    }

    return data;
}

void print(const ai::VTD& data) {
    for(const auto& row : data) {
        for(const auto& element : row) {
            std::cout << element << ", ";
        }
        std::cout << "\n";
    }
}

template <typename T>
double sig(const T z) {
    return 1 / (1 + exp(-z));
}

template <std::size_t numLayers>
void feedForward(const ai::weights<numLayers>& weights, 
    const std::vector<double>& inputs,
    ai::values<numLayers>& values,
    const std::size_t layer)
{
    if(layer < numLayers) {
        std::vector<double> outputs; // holds the ycap values for each layer
        for(std::size_t node = 0; node < weights[layer].size(); ++node) { // for each node in each layer
            double sum = 0;

            for(std::size_t coef = 1; coef < weights[layer][node].size(); ++coef) { // for each weight for each node
                sum += weights[layer][node][coef] * inputs[coef - 1];
            }
            sum += weights[layer][node][0]; // the x0 * w0, or b value
            outputs.push_back(sig(sum));
        }
        values.setColumn(layer, outputs); // store each y vector in the values matrix
        feedForward(weights, outputs, values, layer + 1);
    }
}

int main(int argc, char **argv) {
    const int NUM_LAYERS = 2;
    const int NUM_ITERATIONS = 10000;

    // Initialize with the number of layers, nodes per layer, and weights per node
    // yeesh, this wouldn't work any other way, but, compile time weight holder-ish!
    // each std::pair<> holds the number of nodes in that layer and the number of weights per node
    ai::weights<NUM_LAYERS> weights = std::array<std::pair<std::size_t, std::size_t>, NUM_LAYERS>{{{4, 3}, {1, 5}}};
    ai::VTD data;

    if(argc == 4) {
        std::fstream fin {argv[1]};
        data = dataReader(fin);
        fin.close();
    } else {
        exit(1);
    }

    weights.print();
    print(data);

    ai::values<NUM_LAYERS> values;

    // repeat for desired number of iterations
    for(int k = 0; k < NUM_ITERATIONS; ++k) {
        for(std::size_t i = 0; i < data.size(); ++i) {
            feedForward(weights, data[i], values, NUM_LAYERS);
        }
    }


    return 0;
}