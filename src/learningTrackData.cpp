/*
 * =====================================================================================
 *
 *       Filename:  learningTrackData.cpp
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


int main(void)
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
