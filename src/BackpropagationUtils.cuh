#pragma once

#include <string>
#include <vector>

#include <fstream>

#include "Backpropagation.cuh"


template <class AF, class OutputLayer>
void writeToDotFile(const std::string& dotfilename, const Backpropagation<AF, OutputLayer>& backpropagation);


template <class AF, class OutputLayer>
std::vector<float> getParameterVector(const Backpropagation<AF, OutputLayer>& backpropagation);



#include "BackpropagationUtils_detail.cuh"

