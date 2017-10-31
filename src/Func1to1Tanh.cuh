/*
 * =====================================================================================
 *
 *       Filename:  Func1to1Tanh.cuh
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


class Func1to1Tanh
{
public:
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return tanhf(x);
	}
	//関数の微分
	__host__ __device__
	static float applyDiff(float x)
	{
		float tanh_x = tanhf(x);
		return 1.0f - (tanh_x * tanh_x);
	}
};




