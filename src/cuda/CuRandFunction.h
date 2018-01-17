/*
 * =====================================================================================
 *
 *       Filename:  CuRandFunction.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年02月12日 01時47分46秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "CuRandManager.h"

#include "CuBlasFunction.h"

namespace cuda
{

void setRandomUniform(float min, float max, DeviceVector& v);

void setRandomUniform(float min, float max, DeviceMatrix& m);

}

