/*
 * =====================================================================================
 *
 *       Filename:  CuSolverDnException.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月22日 22時28分20秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include <stdexcept>

#include <string>

namespace cuda
{

class CuSolverDnException : public std::runtime_error
{
public:
	CuSolverDnException(const std::string& what_arg):
		runtime_error(what_arg.c_str())
	{
	}
	CuSolverDnException(const char* what_arg):
		runtime_error(what_arg)
	{
	}
};

}


