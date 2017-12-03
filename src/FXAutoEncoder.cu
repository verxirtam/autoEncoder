
#include "FXAutoEncoder.cuh"


void FXAutoEncoder::getNormarizeInput(DeviceMatrix& normarize_input)
{
	DBAccessor dba(dbFileName);
	dba.setQuery("select strftime('%s', datetime) as epoch, opening from USDJPY_M1 order by epoch;");
	
	std::vector<float> vector_opening;
	
	while(SQLITE_ROW == dba.step_select())
	{
		long long epoch   = dba.getColumnLongLong(0);
		float opening = static_cast<float>(dba.getColumnDouble(1));
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

//学習用のクエリから指定したレコード数分情報取得する
bool FXAutoEncoder::selectRecord(unsigned int record_count, std::vector<float>& output)
{
	for(unsigned int r = 0; r < record_count; r++)
	{
		//1レコード取得
		if(SQLITE_ROW != dbAccessorLearning.step_select())
		{
			//取得に失敗した場合はfalseを返す
			return false;
		}
		//1レコード文のデータ格納
		long long epoch   = dbAccessorLearning.getColumnLongLong(0);
		float opening = static_cast<float>(dbAccessorLearning.getColumnDouble(1));
		output.push_back(opening);
	}
	return true;
}

void FXAutoEncoder::init
	(
		const std::string& db_file_name,
		unsigned int time_length,
		unsigned int layer_size,
		unsigned int minibatch_size
	)
{
	
	trainingTimeBegin = TimeUtil::stringToEpoch("2017/07/22 00:00:00");
	trainingTimeEnd   = TimeUtil::stringToEpoch("2017/09/01 00:00:00");
	testTimeBegin     = TimeUtil::stringToEpoch("2017/09/01 00:00:00");
	testTimeEnd       = TimeUtil::stringToEpoch("2017/09/22 00:00:00");
	
	dbFileName = db_file_name;
	timeLength = time_length;
	
	dbAccessorLearning.open(dbFileName);
	
	
	DeviceMatrix normarize_input;
	getNormarizeInput(normarize_input);
	
	autoEncoder.init(normarize_input, layer_size, minibatch_size);
	
	dbAccessorLearning.setQuery("select strftime('%s', datetime) as epoch, opening from USDJPY_M1 order by epoch;");
	//キャッシュの初期化
	learningQueryCache.clear();
	//キャッシュ用のデータをクエリから取得
	selectRecord(timeLength - 1, learningQueryCache);
}

bool FXAutoEncoder::learning()
{
	//sqlで取得するレコード数
	unsigned int minibatch_size = autoEncoder.getBackpropagation().getMiniBatchSize();
	unsigned int record_count   = minibatch_size;
	
	//ミニバッチ1つ分のデータを格納するvector
	std::vector<float> vector_opening;
	
	//ミニバッチ1つ分のデータを取得
	//先頭のデータをキャッシュから取得
	vector_opening.insert(vector_opening.begin(), learningQueryCache.begin(), learningQueryCache.end());
	//後半のデータをクエリから取得
	bool r = selectRecord(record_count, vector_opening);
	//selectRecordに失敗した場合はfalseを返す
	if(r == false)
	{
		return false;
	}
	
	//キャッシュを更新する
	learningQueryCache.clear();
	auto copy_begin = vector_opening.end();
	copy_begin -= timeLength - 1;
	auto copy_end = vector_opening.end();
	learningQueryCache.insert(learningQueryCache.end(), copy_begin, copy_end);
	
	//ミニバッチの初期化用のvector
	std::vector<float> input_data(timeLength * minibatch_size, 0.0f);
	
	//取得したデータをDeviceMatrixに格納する
	for(unsigned int i = 0; i < minibatch_size; i++)
	{
		for(unsigned int j = 0; j < timeLength; j++)
		{
			input_data[i * timeLength + j] = vector_opening[i + j];
		}
	}
	//ミニバッチ
	DeviceMatrix input_data_d(timeLength, minibatch_size);
	input_data_d.set(input_data);
	
	//学習を実行する
	autoEncoder.learning(input_data_d);
	
	//実行が正常に終了した場合はTrueを返す
	return true;
}

