'''
分布式计算均值和方差的 reducer
'''
import sys
import numpy as np

def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]
cum_val = 0.0
cum_sumsq = 0.0
cum_n = 0.0

for instance in mapperOut:
    nj = float(instance[0])
    cum_n += nj
    cum_val += nj*float(instance[1])
    cum_sumsq += nj*float(instance[2])

mean = cum_val / cum_n
var_sum = (cum_sumsq - 2 * mean * cum_val + cum_n * mean * mean) / cum_n
print('{}\t{}\t{}'.format(cum_n, mean, var_sum))
sys.stderr.write('report : still alive')