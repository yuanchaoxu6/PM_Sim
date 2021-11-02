#!/bin/bash

for ((i=1;i<=16;i=i*2))
do 
    export OMP_NUM_THREADS=$i
    ./a.out > $i
done

for ((i=32;i<=160;i=i+16))
do
    export OMP_NUM_THREADS=$i
    ./a.out > $i
done

./process
