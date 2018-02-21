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

template <class PerceptronType>
std::vector<float> getParameterVector(const PerceptronType& perceptron);

template <>
template <class In, class Inter, class Out>
std::vector<float> getParameterVector<Perceptron<In, Inter, Out> >(const Perceptron<In, Inter, Out>& perceptron);

template <class AF, class UM>
std::vector<float> getParameterVector<Layer<AF, UM> >(const Layer<AF, UM>& perceptron);

}

#include "BackpropagationUtils_detail.cuh"

