#!/bin/sh
#
echo " "
echo " The sampler script is executed with input read from $1. "
echo " Number of samples requested: $3; the output will be written to $2"
echo " "
#
python sampler.py $1 $2 $3
#python sampler_II.py $1 $2 $3
if [ $? -ne 0 ]; then
    echo " "
    echo "Errors running sampler.py!"
    echo " "
    exit
fi
#
echo " "
echo " Execution completed. Good Bye! "
echo " "
