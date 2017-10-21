
#include "ActivateFunction.cuh"

#include "Tanh.cuh"
#include "ReLU.cuh"

void test()
{
	ActivateFunction<Tanh> af_tanh;
	ActivateFunction<ReLU> af_relu;
	DeviceMatrix x = DeviceMatrix::get0Matrix(2,2);
	DeviceMatrix y = DeviceMatrix::get1Matrix(2,2);
	af_tanh.activate(x, y);
	af_relu.activate(x, y);
}
