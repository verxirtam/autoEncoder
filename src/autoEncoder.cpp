/*
 * =====================================================================================
 *
 *       Filename:  autoEncoder.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月12日 04時47分47秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <ctime>
#include <iostream>

#include <array>
#include <vector>

#include <numeric>

#include <DBAccessor.h>

#include "CuBlasFunction.h"



using track = std::array<float, 4>;

void getMean(const std::vector<track>& v, track& mean)
{
	auto l = [](const track& a, const track& b)
	{
		track ret;
		ret[0] = a[0] + b[0];
		ret[1] = a[1] + b[1];
		ret[2] = a[2] + b[2];
		ret[3] = a[3] + b[3];
		return ret;
	};
	
	track init{0.0f, 0.0f, 0.0f, 0.0f};
	
	mean = std::accumulate(v.begin(), v.end(), init, l);
	float v_size = static_cast<float>(v.size());
	
	mean[0] /= v_size;
	mean[1] /= v_size;
	mean[2] /= v_size;
	mean[3] /= v_size;

}


int testDBAccessor(void)
{
	DBAccessor dba("../../db/TrackData/TrackData_20161124.db");
	
	dba.setQuery("select latitude, longitude, altitude, time from TrackData;");
	
	std::vector<track> v;
	
	while(SQLITE_ROW == dba.step_select())
	{
		double la=dba.getColumnDouble(0);
		double lo=dba.getColumnDouble(1);
		int a=dba.getColumnInt(2);
		long long time=dba.getColumnLongLong(3);
		
		track t
		{
			static_cast<float>(la  ),
			static_cast<float>(lo  ),
			static_cast<float>(a   ),
			static_cast<float>(time)
		};
		
		v.push_back(t);
	}
	
	track mean;
	getMean(v,mean);
	
	std::cout << "mean : ";
	std::cout << mean[0] << '\t';
	std::cout << mean[1] << '\t';
	std::cout << mean[2] << '\t';
	std::cout << mean[3] << '\t';
	std::cout << std::endl;
	
	return 0;
}

int testCUBLAS()
{
	//cudaError_t cuda_error;
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate_v2(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "CUBLAS initialization failed" << std::endl;
		return EXIT_FAILURE;
	}
	
	DBAccessor dba("../../db/TrackData/TrackData_20161124.db");
	
	dba.setQuery("select latitude, longitude, altitude, time from TrackData;");
	
	std::vector<float> v;
	
	while(SQLITE_ROW == dba.step_select())
	{
		double la=dba.getColumnDouble(0);
		double lo=dba.getColumnDouble(1);
		int a=dba.getColumnInt(2);
		long long time=dba.getColumnLongLong(3);
		
		v.push_back(static_cast<float>(la  ));
		v.push_back(static_cast<float>(lo  ));
		v.push_back(static_cast<float>(a   ));
		v.push_back(static_cast<float>(time));
		//if(v.size() >= 400)
		//{
		//	break;
		//}
	}
	int M = 4;
	int N = v.size() / M;
	
	std::cout << "M = " << M << ", N = " << N << std::endl;
	
	std::vector<float> vec1;
	for(int i = 0; i < N; i++)
	{
		vec1.push_back(1.0f);
	}
	
	std::vector<float> result;
	for(int i = 0; i < M; i++)
	{
		result.push_back(3.0f);
	}
	
	
	
	float* v_d;
	cudaMalloc((void**)&v_d, M*N*sizeof(float));
	stat = cublasSetMatrix(M, N, sizeof(float), v.data(), M, v_d, M);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSetMatrix()" << std::endl;
		return EXIT_FAILURE;
	}
	
	float* vec1_d;
	cudaMalloc((void**)&vec1_d, N*sizeof(float));
	stat = cublasSetVector(N, sizeof(float), vec1.data(), 1, vec1_d, 1);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSetVector()" << std::endl;
		return EXIT_FAILURE;
	}
	
	float* result_d;
	cudaMalloc((void**)&result_d, M*sizeof(float));
	stat = cublasSetVector(M, sizeof(float), result.data(), 1, result_d, 1);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSetVector()" << std::endl;
		return EXIT_FAILURE;
	}
	
	float alpha = 1.0f / static_cast<float>(N);
	float beta = 0.0f;
	stat = cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, v_d, M, vec1_d, 1, &beta, result_d, 1);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSgemv()" << std::endl;
		return EXIT_FAILURE;
	}
	
	stat = cublasGetVector(M, sizeof(float), result_d, 1, result.data(), 1);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasGetVector()" << std::endl;
		return EXIT_FAILURE;
	}
	
	for(int i = 0; i < M; i++)
	{
		std::cout << result[i] << '\t';
	}
	std::cout << std::endl;
	
	
	
	cublasDestroy(handle);
	return EXIT_SUCCESS;
}

int testCUBLASClass()
{
	
	DBAccessor dba("../../db/TrackData/TrackData_20161124.db");
	
	dba.setQuery("select latitude, longitude, altitude, time from TrackData;");
	
	std::vector<float> v;
	
	while(SQLITE_ROW == dba.step_select())
	{
		double la=dba.getColumnDouble(0);
		double lo=dba.getColumnDouble(1);
		int a=dba.getColumnInt(2);
		long long time=dba.getColumnLongLong(3);
		
		v.push_back(static_cast<float>(la  ));
		v.push_back(static_cast<float>(lo  ));
		v.push_back(static_cast<float>(a   ));
		v.push_back(static_cast<float>(time));
		//if(v.size() >= 400)
		//{
		//	break;
		//}
	}
	int M = 4;
	int N = v.size() / M;
	
	std::cout << "M = " << M << ", N = " << N << std::endl;
	
	std::vector<float> vec1(N, 1.0f);
	
	std::vector<float> result(M, 3.0f);
	
	
	DeviceMatrix v_d(M, N);
	v_d.set(v.data());
	
	DeviceVector vec1_d(N);
	vec1_d.set(vec1.data());
	
	DeviceVector result_d(M);
	result_d.set(result.data());
	
	float alpha = 1.0f / static_cast<float>(N);
	float beta = 0.0f;
	Sgemv(&alpha, CUBLAS_OP_N, v_d, vec1_d, &beta, result_d);
	
	result_d.get(result.data());
	
	for(int i = 0; i < M; i++)
	{
		std::cout << result[i] << '\t';
	}
	std::cout << std::endl;
	
	
	
	return EXIT_SUCCESS;
}

int testDeviceVector(void)
{
	using namespace std;
	vector<DeviceVector> dv;
	DeviceVector x(3);
	float a[]={1.0f, 2.0f, 3.0f};
	float b[]={2.0f, 3.0f, 4.0f};
	float c[]={0.0f, 0.0f, 0.0f};
	
	x.set(a);
	dv.push_back(x);
	
	x.set(b);
	dv.push_back(x);
	
	dv[0].get(c);
	cout << "c[] = {";
	cout << c[0] << '\t';
	cout << c[1] << '\t';
	cout << c[2] << "}" << endl;
	
	dv[1].get(c);
	cout << "c[] = {";
	cout << c[0] << '\t';
	cout << c[1] << '\t';
	cout << c[2] << "}" << endl;
	
	vector<DeviceVector> dv1(dv);
	
	dv1[0].get(c);
	cout << "c[] = {";
	cout << c[0] << '\t';
	cout << c[1] << '\t';
	cout << c[2] << "}" << endl;
	
	
	
	return EXIT_SUCCESS;
}


int main(void)
{
	//testCUBLAS();
	//testCUBLASClass();
	return testDeviceVector();
}

