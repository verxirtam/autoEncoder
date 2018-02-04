
#pragma once

#include "TwoLayerPerceptron.cuh"
#include "ActivateMethodElementWise.cuh"
#include "UpdateMethodMomentum.h"

namespace nn
{

template <class ActivateFunction>
using TwoLayerPerceptronInternal = TwoLayerPerceptron<ActivateMethodElementWise<ActivateFunction>, UpdateMethodMomentum>;


}

