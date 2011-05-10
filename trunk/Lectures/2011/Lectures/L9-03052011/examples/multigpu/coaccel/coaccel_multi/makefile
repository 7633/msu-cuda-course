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

all: coaccel_multi.o pattern2d.o
	gcc coaccel_multi.o pattern2d.o -o coaccel_multi -lpthread -L/opt/coaccel/lib64 -lcoaccel -L/opt/cuda/lib64 -lcudart -lm

coaccel_multi.o: coaccel_multi.c
	gcc -g -std=c99 -I/opt/coaccel/include -I/opt/cuda/include -c $< -o $@

pattern2d.o: pattern2d.cu
	nvcc -g -c $< -o $@

clean:
	rm -rf *.o coaccel_multi

