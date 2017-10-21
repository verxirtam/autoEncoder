/*
 * =====================================================================================
 *
 *       Filename:  ReLU.cuh
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


class ReLU
{
public:
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return (x < 0.0f) ? (0.0f) : (x);
	}
	//関数の微分
	__host__ __device__
	static float applyDiff(float x)
	{
		return (x < 0.0f) ? (0.0f) : (1.0f);
	}
};




