#!/bin/bash
set -e

module switch lang/tcsds-1.2.37

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

export OMP_NUM_THREADS=48

make clean
make 
./benchmark
