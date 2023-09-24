#!/bin/sh
#JSUB -q normal
#JSUB -n 4
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job
source /apps/software/anaconda3/bin/activate zkj
python main.py > GraphU-Net/GraphU-Net.txt
