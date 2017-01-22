/*
 * =====================================================================================
 *
 *       Filename:  CuSolverDnManager.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月23日 01時18分31秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "CuSolverDnManager.h"

const char* CuSolverDnManager::getErrorString(cusolverStatus_t status)
{
	switch (status)
	{
		case CUSOLVER_STATUS_SUCCESS:
			return "CUSOLVER_STATUS_SUCCESS";
		case CUSOLVER_STATUS_NOT_INITIALIZED:
			return "CUSOLVER_STATUS_NOT_INITIALIZED";
		case CUSOLVER_STATUS_ALLOC_FAILED:
			return "CUSOLVER_STATUS_ALLOC_FAILED";
		case CUSOLVER_STATUS_INVALID_VALUE:
			return "CUSOLVER_STATUS_INVALID_VALUE";
		case CUSOLVER_STATUS_ARCH_MISMATCH:
			return "CUSOLVER_STATUS_ARCH_MISMATCH";
		case CUSOLVER_STATUS_MAPPING_ERROR:
			return "CUSOLVER_STATUS_MAPPING_ERROR";
		case CUSOLVER_STATUS_EXECUTION_FAILED:
			return "CUSOLVER_STATUS_EXECUTION_FAILED";
		case CUSOLVER_STATUS_INTERNAL_ERROR:
			return "CUSOLVER_STATUS_INTERNAL_ERROR";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
		case CUSOLVER_STATUS_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_NOT_SUPPORTED";
		case CUSOLVER_STATUS_ZERO_PIVOT:
			return "CUSOLVER_STATUS_ZERO_PIVOT";
		case CUSOLVER_STATUS_INVALID_LICENSE:
			return "CUSOLVER_STATUS_INVALID_LICENSE";
	}
	return "<unknown status>";
}


