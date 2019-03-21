#CC = gcc
#CFLAGS = -O2 -Wall -std=c99

#xiangqimoveslib: xiangqimoveslib.o
#	$(CC) -shared -Wl,-soname,xiangqimoveslib -o xiangqimoveslib.so -fPIC ccc.o

CFLAGS := -fPIC -O3 -Wall -Werror -std=c99
CC := gcc
NAME := xiangqimoves

lib: lib$(NAME).so

lib$(NAME).so: $(NAME).o
	$(CC) -shared -Wl,-soname,lib$(NAME).so.$(MAJOR) $^ -o $@

clean:
	$(RM) *.o *.so*
