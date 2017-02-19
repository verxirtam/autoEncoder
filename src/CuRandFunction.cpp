/*
 * =====================================================================================
 *
 *       Filename:  CuRandFunction.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年02月12日 01時52分31秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "CuRandFunction.h"

void setRandomUniform(float min, float max, DeviceVector& v)
{
	unsigned int N = v.getDimension();
	DeviceVector x(N);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), x.getAddress(), N));
	
	float alpha = - (max - min);
	v = DeviceVector::getAlphaVector(N, max);
	Saxpy(&alpha, x, v);
}

void setRandomUniform(float min, float max, DeviceMatrix& m)
{
	unsigned int M = m.getRowCount();
	unsigned int N = m.getColumnCount();
	
	DeviceMatrix m1(M, N);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), m1.getAddress(), M * N));
	
	float alpha = - (max - min);
	m = DeviceMatrix::getAlphaMatrix(M, N, max);
	Saxpy(&alpha, m1, m);
}


