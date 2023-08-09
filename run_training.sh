#!/bin/bash

models=`ls Models/*.h5`

for model in $models 
do
    name=`echo $model | cut -d"/" -f2 | cut -d"." -f1` >> /dev/null 
    file="Results/$name.out"

    touch $file

    echo "Processing "$name"" >> $file | echo "" >> $file

    nohup python3 Handwriting_scripts/train_model.py --gpu 1 --verbose 2 --epochs 30 --model $model &>> $file &
    curpid=$!

    while ps -c $curpid >> /dev/null
    do 
        sleep 120

    done

    echo "" >> $file | echo "Done "$name"" >> $file

done
