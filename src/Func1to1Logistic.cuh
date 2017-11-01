/*
 * =====================================================================================
 *
 *       Filename:  Func1to1Logistic.cuh
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年10月15日 23時03分50秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#pragma once


class Func1to1Logistic
{
public:
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return 1.0f / (1.0f + expf(- x));
	}
	//関数の微分
	__host__ __device__
	static float applyDiff(float x)
	{
		float f_x = 1.0f / (1.0f + expf(- x));
		return f_x * (1.0f - f_x);
	}
};




