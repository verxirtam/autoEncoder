/*
 * =====================================================================================
 *
 *       Filename:  Func1to1Exp.cuh
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


class Func1to1Exp
{
public:
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return expf(x);
	}
	//関数の微分
	__host__ __device__
	static float applyDiff(float x)
	{
		return expf(x);
	}
};




