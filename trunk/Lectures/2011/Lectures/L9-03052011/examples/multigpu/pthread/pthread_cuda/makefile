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

all: pthread_cuda.o pattern2d.o
	gcc pthread_cuda.o pattern2d.o -o pthread_cuda -lpthread -L/opt/cuda/lib64 -lcudart -lm

pthread_cuda.o: pthread_cuda.c
	gcc -g -std=c99 -I/opt/cuda/include -c $< -o $@

pattern2d.o: pattern2d.cu
	nvcc -g -c $< -o $@

clean:
	rm -rf *.o pthread_cuda

