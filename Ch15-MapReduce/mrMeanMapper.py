'''
分布式计算均值和方差的 Mapper
'''
import sys
import numpy as np

def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
input = [float(line) for line in input]
num_inputs = len(input)
input = np.array(input)
sq_input = np.power(input, 2)
print('{}\t{}\t{}'.format(num_inputs, np.mean(input), np.mean(sq_input)))
sys.stderr.write('report:still alive')