#!/bin/bash
rm ./results/with_tf_option.txt
rm ./results/without_tf_option.txt

for i in `seq 1 10`
do
    python main.py 0
done

for i in `seq 1 10`
do
    python main.py 1
done

python eval.py