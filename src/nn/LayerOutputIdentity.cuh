
#pragma once

#include "Layer.cuh"
#include "ActivateMethodOutputIdentity.cuh"
#include "UpdateMethodMomentum.h"

namespace nn
{

using LayerOutputIdentity = Layer<ActivateMethodOutputIdentity, UpdateMethodMomentum>;


}

