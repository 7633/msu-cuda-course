##
## MSU CUDA Course Examples and Exercises.
##
## Copyright (c) 2011 Dmitry Mikushin
##
## This software is provided 'as-is', without any express or implied warranty.
## In no event will the authors be held liable for any damages arising 
## from the use of this software.
## Permission is granted to anyone to use this software for any purpose, 
## including commercial applications, and to alter it and redistribute it freely,
## without any restrictons.
##

all: boost_cuda.o pattern2d.o
	g++ boost_cuda.o pattern2d.o -o boost_cuda -L/opt/cuda/lib64 -lcuda -lcudart -lm -lboost_thread-mt

boost_cuda.o: boost_cuda.cpp boost_cuda.h
	g++ -g -I/opt/cuda/include -c $< -o $@

pattern2d.o: pattern2d.cu
	nvcc -g -c $< -o $@

clean:
	rm -rf *.o boost_cuda

