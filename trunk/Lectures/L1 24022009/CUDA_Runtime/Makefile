#
# Makefile to build CUDA Hello World for Linux
#
# Author: Alex V. Boreskoff, <steps3d@narod.ru>, <steps3d@gmail.com>
#


all:  hello

hello: hello.cu
	nvcc hello.cu -o $@

clean:
	rm -r -f hello *.o $(OBJS) 2> /dev/null


