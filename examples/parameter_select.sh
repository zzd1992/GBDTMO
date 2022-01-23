#!/bin/bash

for lr in 0.1 0.2
do 
    for depth in 4 5
    do
        echo $lr $depth
        python3 test.py -lr $lr -depth $depth
    done
done
