/*
 * =====================================================================================
 *
 *       Filename:  Tanh.h
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

#include "DeviceMatrix.h"

class Tanh
{
public:
	//活性化関数
	static DeviceMatrix& activate(const DeviceMatrix& x, DeviceMatrix& y);
	//活性化関数の微分
	static DeviceMatrix& activateDiff(const DeviceMatrix& x, DeviceMatrix& y);
};
