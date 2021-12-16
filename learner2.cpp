#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <cmath>
#include "aitools.h"

constexpr int NUM_LAYERS = 2;
constexpr double ETA = 0.001;
constexpr int NUM_ITERATIONS = 10000;
constexpr std::size_t INPUTS = 2;
constexpr std::size_t OUTPUTS = 1;

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
    ai::values<numLayers+1>& output,
    const std::size_t layer)
{
    if(layer < numLayers) {
        std::vector<double> tempOutputs; // holds the ycap values for each layer
        for(std::size_t node = 0; node < weights[layer].size(); ++node) { // for each node in each layer
            double sum = 0;

            for(std::size_t coef = 1; coef < weights[layer][node].size(); ++coef) { // for each weight for each node
                sum += weights[layer][node][coef] * inputs[coef - 1]; // subtract 1 from coef because we don't store x0
            }
            sum += weights[layer][node][0]; // the x0 * w0, or b value
            tempOutputs.push_back(sig(sum));
        }
        output.setColumn(layer, tempOutputs); // store each y vector in the output matrix
        feedForward(weights, tempOutputs, output, layer - 1);
    }
}

template <std::size_t numLayers>
void backProp(ai::values<numLayers>& delta, 
    const ai::values<numLayers+1>& output, 
    ai::weights<numLayers>& weights, 
    const std::array<double, OUTPUTS> Y)
{
    // compute the output layer first
    std::vector<double> tempDelta;
    // for each node in the output layer
    for(std::size_t q = 0; q < output[0].size(); ++q) {
        tempDelta.push_back((Y[q] - output[0][q]) * output[0][q] * (1.0 - output[0][q]));
    }
    delta.setColumn(0, tempDelta);

    for(std::size_t j = 1; j < NUM_LAYERS; ++j) {
        tempDelta.clear();
        double d = 0;
        for(std::size_t q = 0; q < output[j-1].size(); ++q) {
            for(std::size_t a = 0; a < weights[j-1][q].size(); ++a) {
                d += delta[j-1][q] *  weights[j-1][q][a];
            }
        }
        for(std::size_t q = 0; q < output[j].size(); ++q) {
            tempDelta.push_back(d * output[j][q] * (1.0 - output[j][q]));
        }
        delta.setColumn(j, tempDelta);
    }

    for(std::size_t j = 0; j < NUM_LAYERS; ++j) {
        for(std::size_t q = 0; q < output[j].size(); ++q) {
            for(std::size_t p = 0; p < weights[j][q].size(); ++p) {
                weights[j][q][p] = weights[j][q][p] + ETA * delta[j][q] * output[j+1][p];
            }
        }
    }
}

template <std::size_t numLayers>
double SSE(const ai::weights<numLayers>& weights, 
    const ai::VTD& data, 
    ai::values<numLayers+1>& output)
{
    double ycap;
    double sse = 0;

    // for each e in the set E
    for(std::size_t k = 0; k < data.size(); ++k) {
        feedForward(weights, data[k], output, NUM_LAYERS - 1);
        for(std::size_t i = 0; i < OUTPUTS; ++i) {
            ycap = sig(output[0][i]);
            sse += std::pow(data[k][INPUTS + i] - ycap, 2);
        }
    }

    return sse;
}

int main(int argc, char **argv) {
    srand(time(0));
    // Initialize with the number of layers, nodes per layer, and weights per node
    // yeesh, this wouldn't work any other way, but, compile time weight holder-ish!
    // each std::pair<> holds the number of nodes in that layer and the number of weights per node
    // ai::weights<NUM_LAYERS> weights = std::array<std::pair<std::size_t, std::size_t>, NUM_LAYERS>{{{4, 3}, {1, 5}}};
    ai::weights<NUM_LAYERS> weights = std::array<std::pair<std::size_t, std::size_t>, NUM_LAYERS>{{{1, 5}, {4, 3}}};
    ai::VTD data;

    // open connection to file
    std::string filename = argv[1];
    std::ifstream fin(filename);

    if(argc == 4) {
        data = dataReader(fin);
    } else {
        exit(1);
    }

    fin.close();

    weights.print();
    // print(data);

    ai::values<NUM_LAYERS+1> output;
    ai::values<NUM_LAYERS> delta;
    ai::values<NUM_LAYERS> update;
    std::vector<double> Xvals (INPUTS);
    std::array<double, OUTPUTS> Yvals;

    // repeat for desired number of iterations
    for(int k = 0; k < NUM_ITERATIONS; ++k) {
        for(std::size_t i = 0; i < data.size(); ++i) {
            // feedForward(weights, data[i], output, 0);
            feedForward(weights, data[i], output, NUM_LAYERS - 1);
            for(std::size_t t = 0; t < INPUTS; ++t) {
                Xvals[t] = data[i][t];
            }
            output.setColumn(NUM_LAYERS, Xvals);
            for(std::size_t t = 0; t < OUTPUTS; ++t) {
                Yvals[t] = data[i][t + INPUTS];
            }
            backProp(delta, output, weights, Yvals);
        }
    }

    weights.print();

    // --------------- check weights against validation data -----------------

    filename = argv[2];
    fin.open(filename);

    ai::VTD validData; // stores validation data

    if(fin.good()) {
        validData = dataReader(fin);
    } else {
        std::cout << "Could not open connection to " << argv[2] << std::endl;
        exit(1);
    }

    fin.close();

    double sumOfSquaresError = SSE(weights, validData, output);

    // --------------- output to txt file -----------------

    // open connection to file
    filename = argv[3];
    std::ofstream fout(filename);

    if(fout.good()) {
        // output final weights
        for(std::size_t j = 0; j < NUM_LAYERS; ++j) {
            for(std::size_t q = 0; q < weights[NUM_LAYERS - j - 1].size(); ++q) {
                for(std::size_t a = 0; a < weights[NUM_LAYERS - j - 1][q].size(); ++a) {
                    fout << weights[NUM_LAYERS-j-1][q][a] << " ";
                }
                fout << "\n";
            }
        }

        // output sum-of-squares error
        fout << sumOfSquaresError << "\n\n";

        // output name and class
        fout << "CS-5001: HW#2\n"
             << "Programmer: Noah E. Klein\n\n";

        // output parameters
        fout << "TRAINING:\n"
             << "Using learning rate eta = " << ETA << "\n"
             << "Using " << NUM_ITERATIONS << " iterations.\n\n";

        fout << "LEARNED:\n";
        fout << "Input Units:\n";
        for(std::size_t q = 0; q < weights[1].size(); ++q) {
            for(std::size_t a = 0; a < weights[1][q].size(); ++a) {
                fout << weights[1][q][a] << " ";
            }
            fout << "\n";
        }
        fout << "Output Unit:\n";
        for(std::size_t q = 0; q < weights[0].size(); ++q) {
            for(std::size_t a = 0; a < weights[0][q].size(); ++a) {
                fout << weights[0][q][a] << " ";
            }
            fout << "\n";
        }

        fout << "VALIDATION\n"
             << "Sum-of-Squares Error = " << sumOfSquaresError << std::endl;
    } else {
        std::cout << "Could not open connection to " << argv[3] << std::endl;
        exit(1);
    }

    fout.close();

    return 0;
}