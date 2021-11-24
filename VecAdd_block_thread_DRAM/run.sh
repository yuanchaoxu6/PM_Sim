#!/bin/bash

rm 1 2 4 8 16 32 48 64 80 96 112 128 144 160
sed -i "s/num_threads(160)/num_threads(1)/g"  main.cpp
g++ -fopenmp -O3 main.cpp
sudo ./a.out > 1
last=1;
for ((i=2;i<=16;i=i*2))
do 
    sed -i "s/num_threads($last)/num_threads($i)/g"  main.cpp
    #sed -i "s/omp_get_num_threads($i)/omp_get_num_threads()/g"  main.cpp
    g++ -fopenmp -O3 main.cpp
    last=$i
    sudo ./a.out > $i
done

for ((i=32;i<=160;i=i+16))
do
    sed -i "s/num_threads($last)/num_threads($i)/g"  main.cpp
    #sed -i "s/omp_get_num_threads($i)/omp_get_num_threads()/g"  main.cpp
    g++ -fopenmp -O3 main.cpp
    last=$i
    sudo ./a.out > $i
done

./process
