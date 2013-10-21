CC=gcc
OPT=-O -Wall -Wno-unused -Wno-attributes

all: speedtest 

speedtest: sleefsimddp.c sleefsimdsp.c helperavx.h speedtest.c
	$(CC) $(OPT) -DENABLE_AVX -mavx speedtest.c sleefsimddp.c sleefsimdsp.c -o speedtest -lm

clean :
	rm -f *~ *.o *.s speedtest
