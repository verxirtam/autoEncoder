
#pragma once

#include <string>
#include <vector>

#include <fstream>

#include "Backpropagation.cuh"

namespace
{
	//using namespace cuda;
	
	void writeSingleNodeSetting(std::ostream& out, unsigned int layer_index, unsigned int dimension, const std::string& node_name)
	{
		for(unsigned int j = 0; j < dimension; j++)
		{
			out << node_name << layer_index << "_" << j;
			out << " [shape=circle];";
			out << std::endl;
		}
	}
	
	void writeBiasSetting(std::ostream& out, unsigned int layer_index, const std::string& node_name)
	{
		out << node_name << layer_index;
		out << " [shape=box];";
		out << std::endl;
	}
	
	void writeNodeSettings(std::ostream& out, const std::vector<cuda::DeviceMatrix>& weight)
	{
		unsigned int layer_count = weight.size();
		
		for(unsigned int i = 0; i < layer_count - 1; i++)
		{
			unsigned int dimension = weight[i + 1].getColumnCount();
			writeSingleNodeSetting(out, i, dimension, "node");
			writeBiasSetting(out, i + 1, "bias");
			
		}
		unsigned int dimension = weight[layer_count - 1].getRowCount();
		writeSingleNodeSetting(out, layer_count - 1, dimension, "node");
		
	}
	
	void writeEdgeSettings
		(
			std::ostream& out, 
			const std::vector<cuda::DeviceMatrix>& weight, 
			const std::vector<cuda::DeviceVector>& bias
		)
	{
		unsigned int layer_count = weight.size();
		
		for(unsigned int l = 0; l < layer_count - 1; l++)
		{
			//レイヤlとレイヤl + 1の間のedge(weight[l + 1])を設定する
			auto&& w = weight[l + 1];
			unsigned int from_count = w.getColumnCount();
			unsigned int to_count   = w.getRowCount();
			std::vector<float> weight_vector = w.get();
			for(unsigned int i = 0; i < from_count; i++)
			{
				for(unsigned int j = 0; j < to_count; j++)
				{
					//w[i * to_count + j]
					
					float value = weight_vector[i * to_count + j];
					std::string color = (value >= 0) ? ("blue") : "red";
					unsigned int penwidth = std::min(10, static_cast<int>(std::fabs(value) * 9 + 1));
					
					out << "node" << l     << "_" << i;
					out << " -> ";
					out << "node" << l + 1 << "_" << j;
					out << "[";
					//out << "label = \""  << value << "\",";
					out << "color = "    << color << ",";
					out << "penwidth = " << penwidth;
					out << "];";
					out << std::endl;
				}
			}
			//レイヤl+1のバイアス(bias[l + 1])を設定する
			auto&& b = bias[l + 1];
			std::vector<float> bias_vector = b.get();
			for(unsigned int j = 0; j < to_count; j++)
			{
				float value = bias_vector[j];
				std::string color = (value >= 0) ? ("blue") : "red";
				unsigned int penwidth = std::min(10, static_cast<int>(std::fabs(value) * 9 + 1));
				
				out << "bias" << l + 1;
				out << " -> ";
				out << "node" << l + 1 << "_" << j;
				out << "[";
				//out << "label = \""  << value << "\",";
				out << "color = "    << color << ",";
				out << "penwidth = " << penwidth;
				out << "];";
				out << std::endl;
			}
		}
	}
	void writeRankSettings(std::ostream& out, const std::vector<cuda::DeviceMatrix>& weight)
	{
		unsigned int layer_count = weight.size();
		
		for(unsigned int i = 0; i < layer_count - 1; i++)
		{
			unsigned int dimension = weight[i + 1].getColumnCount();
			out << "{rank = same; ";
			for(unsigned int j = 0; j < dimension; j++)
			{
				out << "node" << i << "_" << j << "; ";
			}
			out << "bias" << i + 1 << "; ";
			out << "}" << std::endl;
			
		}
		out << "{rank = same; ";
		unsigned int dimension = weight[layer_count - 1].getRowCount();
		for(unsigned int j = 0; j < dimension; j++)
		{
			out << "node" << layer_count - 1 << "_" << j << "; ";
		}
		out << "}" << std::endl;
		
	}
}

template<class AF, class OutputLayer>
void writeToDotFile(const std::string& dotfilename, const Backpropagation<AF, OutputLayer>& backpropagation)
{
	auto weight = backpropagation.getWeight();
	auto bias   = backpropagation.getBias();
	
	
	
	std::ofstream dotfile(dotfilename);
	dotfile << "digraph {" << std::endl;
	dotfile << "graph[rank_dir = LR, nodesep=0.5, ranksep=3.0];" << std::endl;
	
	writeNodeSettings(dotfile, weight);
	writeEdgeSettings(dotfile, weight, bias);
	writeRankSettings(dotfile, weight);
	
	dotfile << "}" << std::endl;
}




template <class AF, class OutputLayer>
std::vector<float> getParameterVector(const Backpropagation<AF, OutputLayer>& backpropagation)
{
	//TODO host <-> device間の通信を減らしたい
	
	std::vector<float> result;
	
	unsigned int layer_count = backpropagation.getLayerCount();
	auto w = backpropagation.getWeight();
	auto b = backpropagation.getBias();
	
	for(unsigned int l = 1; l < layer_count; l++)
	{
		auto w_l = w[l].get();
		auto b_l = b[l].get();
		result.insert(result.end(), w_l.begin(), w_l.end());
		for(int i = 0; i < 10; i++)
		{
			result.push_back(-1.0f);
		}
		result.insert(result.end(), b_l.begin(), b_l.end());
		for(int i = 0; i < 10; i++)
		{
			result.push_back( 1.0f);
		}
	}
	
	return result;
}


