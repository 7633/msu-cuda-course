#
# Makefile to build CUDA examples for Linux
#
# Author: Alex V. Boreskoff, <steps3d@narod.ru>, <steps3d@gmail.com>
#

include ../make.inc

all:  info matmult1 matmult2

info: info.cu
	nvcc info.cu -o $@

matmult1: matmult1.cu
	nvcc matmult1.cu -o $@

matmult2: matmult2.cu
	nvcc matmult2.cu -o $@

clean:
	rm -r -f info matmult1 matmult2 *.o $(OBJS) 2> /dev/null


include ../rules.linux
