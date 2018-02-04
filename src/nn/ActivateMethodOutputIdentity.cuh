/*
 * =====================================================================================
 *
 *       Filename:  ActivateMethodOutputIdentity.cuh
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年02月05日 04時18分46秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "../cuda/DeviceMatrix.h"

#include "ElementWiseFunction2to1.cuh"
#include "Func2to1ElementWiseDifference.cuh"

namespace nn
{

class ActivateMethodOutputIdentity
{
public:
	static void activate(const cuda::DeviceMatrix& u, cuda::DeviceMatrix& z)
	{
		z = u;
	}
	static void getDelta
		(
			const cuda::DeviceMatrix& u,
			const cuda::DeviceMatrix& z,
			const cuda::DeviceMatrix& d,
			cuda::DeviceMatrix& delta
		)
	{
		// diff : (x, y) -> x - y
		using diff = ElementWiseFunction2to1<Func2to1ElementWiseDifference>;
		
		//delta = diff(z, d)
		//      = z - d
		diff::apply(z, d, delta);
	}
};

}//namespace nn




