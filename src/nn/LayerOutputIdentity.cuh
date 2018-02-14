
#pragma once

#include "Layer.h"
#include "ActivateMethodOutputIdentity.cuh"
#include "UpdateMethodMomentum.h"

namespace nn
{

using LayerOutputIdentity = Layer<ActivateMethodOutputIdentity, UpdateMethodMomentum>;


}

