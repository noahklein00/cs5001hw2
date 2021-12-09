#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using examples = std::vector<std::vector<double>>;

examples dataReader(std::istream& in) {
    constexpr char DELIMITER = '\t';
    std::string curLine;
    examples data;
    std::size_t index = 0;

    while(std::getline(in, curLine) && curLine.size() > 0) {
        std::stringstream unsplitLine(curLine);
        std::string catchData;
        data.push_back({});
        data[index].push_back(1); // x0 is always 1

        while(std::getline(unsplitLine, catchData, DELIMITER)) {
            data[index].push_back(std::stod(catchData));
        }
        ++index;
    }

    return data;
}

int main(int argc, char **argv) {

    return 0;
}