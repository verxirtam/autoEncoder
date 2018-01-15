
#include "Backpropagation.cuh"

#include "Func1to1Tanh.cuh"
#include "OutputLayerRegression.cuh"

namespace nn
{

using BackpropagationTanhReg = Backpropagation<Func1to1Tanh, OutputLayerRegression>;

}


