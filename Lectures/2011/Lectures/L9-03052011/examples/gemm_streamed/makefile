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

NAME = gemm_streamed

COMP = gcc -std=c99 -g -O0

INCLUDES := -I/opt/cuda/include -I/usr/local/cuda/include
LIBPATH := -L/opt/cuda/lib64 -L/usr/local/cuda/lib64
DEPLIBS := -lcublas -lcudart -lm -lblas

all: $(NAME).c $(NAME).h
	$(COMP) $(INCLUDES) $(NAME).c $(LIBPATH) $(DEPLIBS) -o $(NAME)

clean:
	rm -rf $(NAME)

snap:
	tar -cvzf ../$(NAME)_`date +%y%m%d%H%M%S`.tar.gz ../$(NAME)
