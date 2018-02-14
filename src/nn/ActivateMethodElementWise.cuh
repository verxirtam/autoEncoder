/*
 * =====================================================================================
 *
 *       Filename:  ActivateMethodElementWise.cuh
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

#include "ElementWiseFunction1to1.cuh"
#include "ElementWiseFunctionUtil.cuh"

namespace nn
{

template <class ActivateFunction>
class ActivateMethodElementWise
{
public:
	static void activate(const cuda::DeviceMatrix& u, cuda::DeviceMatrix& z)
	{
		//z = ActivateFunction(u)
		ElementWiseFunction1to1<ActivateFunction>::apply(u, z);
	}
	static void getDelta
		(
			const cuda::DeviceMatrix& u,
			const cuda::DeviceMatrix& z,
			const cuda::DeviceMatrix& weight_t_delta,
			cuda::DeviceMatrix& delta
		)
	{
		//Func: (x, y) -> ActivateFunction'(x) ** y
		using Func = Func2to1Composite1st<Func2to1ElementWiseProduct, Func1to1ApplyDiff<ActivateFunction> >;
		
		//delta = Func(u, weight_t_delta)
		//         = ActivateFunction'(u) ** weight_t_delta
		ElementWiseFunction2to1<Func>::apply
			(
				u,
				weight_t_delta,
				delta
			);
	}
};

}//namespace nn




