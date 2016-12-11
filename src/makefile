#C++向けで、以下の前提で使用可能なmakefile
#
#・同一フォルダにヘッダ・ソースと実行ファイルが格納されている
#・実行ファイルはフォルダ名と同一
#・main関数を持つソースはフォルダ名.cpp
#・ソースの拡張子はcpp

#変数定義
###################################################

#ターゲット
#作成する実行ファイル
#2個上のフォルダ名と同一
TARGET := $(shell basename `dirname \`pwd\``)

#テスト用のターゲット
#テストを実行する
TESTTARGET := unittest

#NVCC、g++共通のフラグ（デバッグ出力、最適化レベル）:デバッグ用
COMMON_FLAGS := -pg -g -O0
#COMMON_FLAGS := 

#CXX_LINKS
CXX_LINKS := -lsqlite3

#ホームディレクトリのパス(末尾に"/"無し)
HOME := $(shell echo ~)

#g++のコンパイルのフラグ
CXXFLAGS := -Wall -fopenmp -I/usr/include -I/usr/local/cuda/include -I$(HOME)/programs/ATFViewer/include $(CXX_LINKS)
CXXONLYFLAGS := -std=c++11 -Weffc++

#CUDA関連の設定

#CUDAコンパイラ
NVCC := nvcc

#CUDAコンパイルオプション(-Gはデバッグ情報出力)
NVCC_FLAGS := --std c++11 -G -Wno-deprecated-gpu-targets

#NVCC向けのg++オプション
NVCC_CXX_FLAGS := $(CXXFLAGS:%=-Xcompiler %)

#CUDAリンクオプション
NVCC_LINKS := -lcuda -lcudart $(CXX_LINKS)


#ソースファイル
#フォルダ直下の.cppファイル
SRCS := $(shell ls *.cpp)

#CUDAソース
#フォルダ直下の.cuファイル
CUDA_SRCS := $(shell ls *.cu)

#オブジェクトファイル
#.cpp or .cuを.oに置換している
OBJS := $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

#ターゲット用のオブジェクトファイル
OBJSTARGET := $(filter-out $(TESTTARGET).o $(GLTESTTARGET).o, $(OBJS))

#テスト用のオブジェクトファイル
OBJSTEST := $(filter-out $(TARGET).o $(GLTESTTARGET).o, $(OBJS))

#依存関係ファイル
DEPS := $(SRCS:.cpp=.d) $(CUDA_SRCS:.cu=.d)

#gprof2dot.pyのパス
GPROF2DOT := ~/tools/gprof2dot/gprof2dot.py

#nvprofコマンド
NVPROF := nvprof --csv

#コマンド
###################################################

#全てコンパイルする
#先頭に書いたコマンドはmakeコマンドでオプション無しで実行される
.PHONY: all
all: $(TARGET)
#all: $(TARGET) $(TESTTARGET)

.PHONY: run
run: $(TARGET)
	$(NVPROF) --log-file $(TARGET).nvprof ./$(TARGET)
	if [ -f core ]; then banner core dumped!;fi
	gprof $(TARGET) | $(GPROF2DOT) | dot -Tpng -o gprof_call_graph_$(TARGET).png

#テストの実行ファイルを実行する
#ログと標準出力両方に出力する
#.PHONY: test
#test: $(TESTTARGET)
#	$(NVPROF) --log-file $(TESTTARGET).nvprof ./$(TESTTARGET) 2>&1 | tee -a $(TESTTARGET).log
#	if [ -f core ]; then banner core dumped!;fi
#	gprof $(TESTTARGET) | $(GPROF2DOT) | dot -Tpng -o gprof_call_graph_$(TESTTARGET).png
#	./nvprof_graph.sh $(TESTTARGET)

#githubにアップロードする
.PHONY: git
git: $(TARGET)
	(cd ..; git add --all .;)
	(cd ..; git commit -a;)
	(cd ..; git push -u origin master;)

.PHONY: doc
doc: $(TARGET)
	(cd ..; doxygen;)
	(cd ../docs; git add .;)
	(cd ../docs; git commit -m "update docs.";)
	(cd ../docs; git push origin gh-pages;)
	(cd ..; git add docs/;)
	(cd ..; git commit -m "update docs.";)


#makefile自体のテスト用
MAKETEST_EXIST = glunittest
MAKETEST_NOTEXIST = notexistfile
.PHONY: maketest
maketest:
	echo AAA$(CXXFLAGS:%=-Xcompiler %)AAA
	banner test
	if [ -f $(MAKETEST_EXIST)    ]; then banner "exist."; else banner "not exist.";fi
	if [ -f $(MAKETEST_NOTEXIST) ]; then banner "exist."; else banner "not exist.";fi
	if [ -f core                 ]; then banner core dumped!;fi
	if [ -f $(MAKETEST_EXIST)    ]; then banner core dumped!;fi

#実行ファイルの作成
#オプションは最後に指定する
#先頭に書くとコンパイルが通らない
#<ターゲット>: ターゲットに依存するファイル
#$@ : ターゲット名
#$< : 最初の依存ファイル名(":"の右側の最初(一番左)の項目)
#$^ : 依存ファイル全体
$(TARGET): $(OBJSTARGET)
	$(NVCC) $(NVCC_LINKS) $(COMMON_FLAGS) $(NVCC_FLAGS) -o $@ $^ $(NVCC_CXX_FLAGS)

#テスト用の実行ファイルを作成
#$(TESTTARGET): $(OBJSTEST)
#	$(NVCC) $(NVCC_LINKS) $(COMMON_FLAGS) $(NVCC_FLAGS) -o $@ $^ $(NVCC_CXX_FLAGS)

#オブジェクトファイルの作成
#"-MMD"オプションによって依存関係ファイル(*.d)も同時に作成する
%.o: %.cpp
	$(CXX) -c -MMD $< $(COMMON_FLAGS) $(CXXONLYFLAGS) $(CXXFLAGS)

%.o: %.cu
	$(NVCC) $(COMMON_FLAGS) $(NVCC_LINKS) $(NVCC_FLAGS) $(NVCC_CXX_FLAGS) -c $<
	$(NVCC) $(COMMON_FLAGS) $(NVCC_LINKS) $(NVCC_FLAGS) $(NVCC_CXX_FLAGS) -M $< > $*.d

#実行ファイル・中間ファイルの削除
#".PHONY"はターゲットではなくコマンドであることを明示するためのコマンド
.PHONY: clean
clean:
	$(RM) $(TARGET) $(TESTTARGET) $(GLTESTTARGET) $(OBJS) $(DEPS)

#依存関係ファイルをインクルードする
#先頭に”-”を書いているので依存関係ファイルがなくてもエラーとならない
-include $(DEPS)
