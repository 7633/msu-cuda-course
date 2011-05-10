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

all: mpi_cuda.o pattern2d.o
	mpicc mpi_cuda.o pattern2d.o -o mpi_cuda -lrt -lpthread -L/opt/cuda/lib64 -lcudart -lm

mpi_cuda.o: mpi_cuda.c
	mpicc -g -std=c99 -D_GNU_SOURCE -I/opt/cuda/include -c $< -o $@

pattern2d.o: pattern2d.cu
	nvcc -g -c $< -o $@

clean:
	rm -rf *.o mpi_cuda

