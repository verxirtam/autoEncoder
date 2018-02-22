#pragma once

#include <string>
#include <vector>

#include <fstream>

#include "Backpropagation.cuh"

#include "./Perceptron.cuh"
#include "./Layer.h"

namespace nn
{

template <class AF, class OutputLayer>
void writeToDotFile(const std::string& dotfilename, const Backpropagation<AF, OutputLayer>& backpropagation);


template <class AF, class OutputLayer>
std::vector<float> getParameterVector(const Backpropagation<AF, OutputLayer>& backpropagation);

template <class Input, class Internal, class Output>
std::vector<float> getParameterVector(const Perceptron<Input, Internal, Output>& perceptron);

template <class ActivateFunction, class UpdateMethod>
std::vector<float> getParameterVector(const Layer<ActivateFunction, UpdateMethod>& perceptron);

}

#include "BackpropagationUtils_detail.cuh"

