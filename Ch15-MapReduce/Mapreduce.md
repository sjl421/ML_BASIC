# 大数据和 MapReduce

PS ：机器学习实战最后一章，简单了解 MapReduce 和大数据

## MapReduce

MapReduce 在大量节点组成的集群上运行。它的工作流程是：单个作业被分为很多小份，输入数据也被切片分发到每个节点，各个结点旨在本地数据上做运算，对应的运算代码称为 mapper，这个过程被称作 map 阶段。每个 mapper 的输出通过某种方式组合（一般还会做排序）。排序后的结果再被分成小份分发到各个节点进行下一步处理工作。第二部的处理阶段被称为 reduce 接待你，对应的运行代码被称为 reducer。reducer 的输出就是程序的最终执行结果。

要点：
1. 主节点控制 MapReduce 的作业流程
2. MapReduce 的作业可以分为 map 任务和 reduce 任务
3. map 任务之间不做数据交流，reduce 任务之间也没做数据交流
4. 在 map 和 reduce 阶段中间，有一个 sort 或 combine 阶段
5. 数据被重复放在不同的机器上，以防某个及其失效
6. mapper 和 reducer 传输的数据形式为 key/value 对

## Hadoop 流

Hadoop 项目是 MapReduce 框架的一个实现。下面将使用 Python 编写 MapReduce 代码，并在 Hadoop 流中运行。Hadoop 流的执行像 linux 系统中的管道

`python mapper.py < inputfile.txt | python reducer.py > outfile.txt`

实战：
1. 分布式计算均值和方差的 mapper。代码详见 [mrMeanMapper.py](./mrMeanMapper.py)。通过 `python mrMeanMapper.py < ./dataset/inputFile.txt` 命令执行代码
2. 分布式计算均值和方差的 reducer。代码详见 [mrMeanReducer.py](./mrMeanReducer.py)。通过 `python mrMeanMapper.py < ./dataset/inputFile.txt | python mrMeanReducer.py` 命令执行代码