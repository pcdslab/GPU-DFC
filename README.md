# GPU-DFC
This repository contains Fast-GPU-PCC algorithm

#Compilation
Use the following command for compiling the program

#Running
Use the follwoing command to run the project.

./out N L W S M

##Description of arguments
N: number of regions/voxel

L: length of time series

W: window size

S: step size

M: Method (1,2,3). 1: No memory reduction, 2: Matrix decomposition, 3: sparsification

The data should be stored in row major format (first N elements corresponds to time series of first element, second N elements corresponds to time series of first element and …)
