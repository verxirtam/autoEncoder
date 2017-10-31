
#include "ActivateFunction.cuh"

#include "Func1to1Tanh.cuh"
#include "Func1to1ReLU.cuh"

void test()
{
	ActivateFunction<Func1to1Tanh> af_tanh;
	ActivateFunction<Func1to1ReLU> af_relu;
	DeviceMatrix x = DeviceMatrix::get0Matrix(2,2);
	DeviceMatrix y = DeviceMatrix::get1Matrix(2,2);
	af_tanh.activate(x, y);
	af_relu.activate(x, y);
}
