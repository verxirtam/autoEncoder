
#include "FXAutoEncoderDBAccessor.cuh"

std::vector<float> FXAutoEncoderDBAccessor::getDataAtTime(long long t, unsigned int limit)
{
	std::vector<float> result;
	
	std::stringstream query;
	query << "select ";
	query << "strftime('%s', datetime) as epoch, opening ";
	query << "where ";
	query << "epoch <= " << t << " ";
	query << "from USDJPY_M1 ";
	query << "limit " << limit << " ";
	query << "order by epoch ";
	query << ";";
	
	DBAccessor dba(dbFileName);
	dba.setQuery(query.str());
	
	while(SQLITE_ROW == dba.step_select())
	{
		long long epoch   = dba.getColumnLongLong(0);
		float opening = static_cast<float>(dba.getColumnDouble(1));
		result.push_back(opening);
	}
	
	return result;
}
