
#pragma once

#include "Layer.cuh"
#include "ActivateMethodElementWise.cuh"
#include "UpdateMethodMomentum.h"

namespace nn
{

template <class ActivateFunction>
using LayerInternal = Layer<ActivateMethodElementWise<ActivateFunction>, UpdateMethodMomentum>;


}

