#!/bin/bash

mkdir -p /home/akiva/gpnettrain/logs

python /home/akiva/FCNX-Ensembling/FCN3Exps/ensembling_fcn3.py --P 20 --d 20 --lrA 1e-6 --N 400 --chi 400  \
  > /home/akiva/gpnettrain/logs/P20_D20_lrA1e-6_N400.log 2>&1

python /home/akiva/FCNX-Ensembling/FCN3Exps/ensembling_fcn3.py --P 20 --d 20 --lrA 1e-6 --N 1000 --chi 1000\
  > /home/akiva/gpnettrain/logs/P20_D20_lrA1e-6_N1000.log 2>&1 