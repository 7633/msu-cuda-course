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

all: shmem_mmap.c
	gcc -std=c99 -D_GNU_SOURCE shmem_mmap.c -o shmem_mmap -lrt -lpthread

clean:
	rm -rf shmem_mmap

