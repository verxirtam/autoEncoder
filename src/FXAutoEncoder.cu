
#include "FXAutoEncoder.cuh"


void FXAutoEncoder::getNormarizeInput(DeviceMatrix& normarize_input)
{
	DBAccessor db(dbFileName);
	db.setQuery("select strftime('%s', datetime) as epoch, opening from USDJPY_M1 order by epoch;");
	
	std::vector<float> vector_opening;
	
	while(SQLITE_ROW == db.step_select())
	{
		long long epoch   = db.getColumnLongLong(0);
		float opening = static_cast<float>(db.getColumnDouble(1));
		vector_opening.push_back(opening);
	}
	
	
	std::vector<float> normarize_input_host;
	//正規化用のデータ数
	unsigned int normarize_input_length = 100;
	//データの時間間隔
	unsigned int interval = (vector_opening.size() - timeLength) / normarize_input_length;
	for(unsigned int i = 0; i < normarize_input_length; i++)
	{
		for(unsigned int j = 0; j < timeLength; j++)
		{
			unsigned int n = i * interval + j;
			normarize_input_host.push_back(vector_opening[n]);
		}
	}
	
	normarize_input = DeviceMatrix(timeLength, normarize_input_length, normarize_input_host);
}

void FXAutoEncoder::init
	(
		const std::string& db_file_name,
		unsigned int time_length,
		unsigned int layer_size,
		unsigned int minibatch_size
	)
{
	dbFileName = db_file_name;
	timeLength = time_length;
	
	
	DeviceMatrix normarize_input;
	getNormarizeInput(normarize_input);
	
	autoEncoder.init(normarize_input, layer_size, minibatch_size);
	
}

void FXAutoEncoder::learning()
{
	
	DBAccessor db(dbFileName);
	db.setQuery("select strftime('%s', datetime) as epoch, opening from USDJPY_M1 order by epoch;");
	
	std::vector<float> vector_opening;
	
	while(SQLITE_ROW == db.step_select())
	{
		long long epoch   = db.getColumnLongLong(0);
		float opening = static_cast<float>(db.getColumnDouble(1));
		vector_opening.push_back(opening);
	}
	
	unsigned int minibatch_size = autoEncoder.getBackpropagation().getMiniBatchSize();
	std::vector<float> input_data(timeLength * minibatch_size);
	DeviceMatrix input_data_d(timeLength, minibatch_size);
	
	unsigned int vector_opening_length = vector_opening.size();
	unsigned int input_data_count = vector_opening_length / (timeLength * minibatch_size);
	
	for(unsigned int n = 0; n < input_data_count; n++)
	{
		//nはミニバッチのカウンタ
		
		for(unsigned int i = 0; i < minibatch_size; i++)
		{
			for(unsigned int j = 0; j < timeLength; j++)
			{
				input_data[i * timeLength + j] = vector_opening[n * minibatch_size + i + j];
			}
		}
		input_data_d.set(input_data);
		autoEncoder.learning(input_data_d);
	}
}

