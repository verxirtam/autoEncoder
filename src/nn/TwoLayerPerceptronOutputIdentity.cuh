
#pragma once

#include "TwoLayerPerceptron.cuh"
#include "ActivateMethodOutputIdentity.cuh"
#include "UpdateMethodMomentum.h"

namespace nn
{

using TwoLayerPerceptronOutputIdentity = TwoLayerPerceptron<ActivateMethodOutputIdentity, UpdateMethodMomentum>;


}

