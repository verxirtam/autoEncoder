#pragma once

#include <vector>
#include <sstream>

#include <DBAccessor.h>

class FXAutoEncoderDBAccessor
{
private:
	std::string dbFileName;
public:
	FXAutoEncoderDBAccessor():
		dbFileName("")
	{
	}
	FXAutoEncoderDBAccessor(const std::string& db_file_name):
		dbFileName(db_file_name)
	{
	}
	void setDbFileName(const std::string& db_file_name)
	{
		dbFileName = db_file_name;
	}
	std::vector<float> getDataAtTime(long long t, unsigned int limit);
};
