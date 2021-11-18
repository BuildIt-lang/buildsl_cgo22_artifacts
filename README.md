# BuilDSL Artifact Evaluation

## Introduction
This repository documemts the evaluation procedure for the artifacts of our paper "GraphIt to CUDA compiler in 2021 LOC: A case for high-performance DSL implementation via staging with BuilDSL" which is built on top of [BuildIt](https://buildit.so). The artifacts are divided into 2 sections - 
  - Section 1. Reproduce the results from Figure 11 in the paper that compares the performance of code generated from BuilDSL with the code generated from the state of the art DSL GraphIt. 
  - Section 2. (Optional) Use the Generalized DSL framework presented in the paper to implement a very simple matrix multiplication DSL. 

Since Figure 11 shows the performance numbers when run on the NVIDIA-Tesla V-100 GPU, the exact execution times you will get in Section 1 will depend on the actual GPU you use. If you do not have access to the same GPU, we have provided access to our system with this GPU in our artifact evaluation submission. If you use any other GPU the schedules might have to be tuned to get the best performance for the GPU.

## Hardware and Software requirements
We expect you to run the artifact evaluation on a Linux system with at least 40GBs of space. Following are the software requirements for each of the parts

### Section 1
Since Section 1 actually evaluates the performance of the generated code on the GPU, this section requires access to a GPU. The software requirements are - 
 - cmake (>= 3.5.1)
 - CXX compiler (like g++ >= 7.5.0 or clang equivalent)
 - python3 
 - make
 - bash
 - git
 - CUDA 10.1 or higher

### Section 2
Since Section 2 only implements a new DSL, you don't need access to a GPU. The requirements are the same as above except CUDA and the GPU. 

All the requirements for both the sections are available on the system provided for artifact evaluation. Please refer to the artifact evaluation pdf for instructions on how to connect to our servers. 

## How to run

### Cloning the repository

To start, first clone this repository using the following command - 

```
git clone --recursive https://github.com/BuildIt-lang/buildsl_cgo22_artifacts.git
```

If you have already cloned the repository without the recursive flag, you can run the following commands inside the cloned directory to fetch all the submodules - 

```
git submodules init
git submodules update
```

Now navigate to the main repostitory and continue the rest of the steps. 

### Build all dependencies
We will first build all the dependencies which have been placed in the repository as submodules. This includes the GraphIt GPU compiler repository, the BuilDSL repository and the BuildIt repository inside BuilDSL that it depends on. 

To build all the dependencies, simply run the following command from the top level directory of this repository - 

```
bash build_all_deps.sh
```

This command should take maximum of 10 mins to build all the dependencies. If any of the particular submodules fails to build, it will be reported and you can try to build it again. 


### Fetch the datasets
We have created two datasets for your convenience - *small* and *all*. The small dataset contains just two graphs (one with bounded degree distribution and one with power law degree distribution). Obtaining and running the small dataset should take less than 15 mins and quickly tests all the variants for all algorithms. The all dataset contains all the 9 graphs from the paper and would take much longer to run (upwards of 2 hours on our system). 

There are two ways of obtaining the datasets. If you are running this artifact evaluation on the system we have provided access to, you can quickly fetch all the data set files by running the following commands in the top level directory - 

    cd dataset
    make local
    
If everything succeeds, the dataset should be soft-linked into this directory and you can verify that by running the `ls` command. You can now navigate to the top level directory using the command `cd ../` and proceed to the next step. If the command reports the error 

> You are not running this command on the right host. Please use `make dataset` instead

it means that you are not running the artifact evaluation on our system and you should use the other method for downloading the datasets

If you are running the artifact evaluation on your own system, the script will have to download a tar ball and extact the files. We have a separate command for *small* and *all* datasets. So if you are planning to run the evaluation only for the 2 graphs, please download only the small dataset to save time.

For downloading the *all* dataset run the following commands from the top-level directory -  

    cd datasets
    make dataset

To download just the *small* dataset run the following commands from the top-level directory - 

    cd datasets
    make small

    
This step will take some time, because it downloads and uncompresses all the datasets. After the command succeeds, you can verify that the files are downloaded by running the `ls` command. The small dataset is part of the all dataset and if you accidently downloaded the all dataset, you can still run the small part of the experiment. 

Navigate to the top level directory in any case using the command - `cd ../`


### Section 1: Reproduce Figure 11
The purpose of this evaluation is to run all the 5 applications on all the 9 datasets with both GraphIt and BuilDSL and compare their execution times. Before we actually run the evaluation, we will list all the GPUs in the system and find one that is completely free. We need a free GPU because the performance might be hampered if other processes are running on the same GPU. 

Start by running the command - 

    nvidia-smi
    
This will list all the GPUs attached to the system numbered from 0. At the bottom of the table, there is a Processes section which shows what processes are running on which GPU. Find a GPU which doesn't have any processes running on it and note down its ID. Suppose for the purpose of this evaluation, the 4th GPU (ID: 3) is free and we want to use that. 

We do not recommend running the evaluation on a GPU that is being used by other processes since it might affect the evaluation results (and correctness) a lot. 


To run only the small data set navigate to the top level directory of the repository and run the command - 

    python3 gen_fig11.py small

To run the all data set navigate to the top level directory of the repository and run the command - 

    python3 gen_fig11.py


Immediately this command will prompt a few configuration options (with default options in []). If you are running on our server, all the default values would be appropriate. You can select the default by simply pressing enter. 

The last prompt will ask the GPU index to use. Just type the index you had noted above. The default here will be zero and can be selected by just pressing enter as before. 

If you are running on your own servers, the only options you might have to specify are path to the CXX compiler and the path to the CUDA SDK. The default path for CUDA is `/usr/local/cuda-10.1/bin/nvcc`. You might have it at a different location depending on the version installed. 

The script will then start running all the evaluations. This step will take 1.5-2 hrs depending on the system (for the *all* dataset). 
Once the script completes, it will print a table with the execution times for the 5 applications and 9 datasets for both GraphIt and BuilDSL. The numbers from both the frameworks will be in the same cell separated by a `/`. (BuilDSL/GraphIt). The main aim of this experiment is to show that BuilDSL generates comparative code to the state of the art graph DSL GraphIt. 

You can also view this table later by running the following command from the top-level directory of this repository - 

```
cat outputs/fig11.txt
```

At this time, you can also browse the code generated from BuilDSL in the outputs directory - 

```
less outputs/buildsl_bfs_power.cu
less outputs/buildsl_bfs_road.cu
```

These files show the code generated for two different schedules for the Breadth First Search application. These code snippets are in the supplementary section of our original paper. 

### (Optional) Section 2: Write a simple DSL with BuilDSL
The purpose of our paper is to show that it is easy to write DSLs with staging and BuilDSL. To demonstrate that we will write a simple DSL for matrix multiplication that generates GPU code. 

We have provided the skeleton code for this section in `mm-dsl-skeleton.cpp`. You can start by copying this into a new file with the following command in the top level directory of this repository - 

```
cp mm-dsl-skeleton.cpp mm-dsl.cpp
```

If you do not want to follow all the steps and just view the final results of the DSL, we have also provided the completed file `mm-dsl-finished.cpp`. You can copy that instead and run it directly with - 

```
cp mm-dsl-finished.cpp mm-dsl.cpp
```

In either case, run the DSL code to generate code for an application using the following command - 

```
bash run_dsl.sh
```

If you have run the finished DSL, you can browse the generated code and the code in the cpp and skip rest of the steps. If you have copied the skeleton, we will build the DSL step by step.


**Skip from here if you ran the finished file**
Before we look at the results of the DSL, we will inspect the source file and understand various sections in it. Open the file mm-dsl.cpp in your favorite editor. We have `vim` and `nano` installed on our server.

At the top of the file, we see all the BuilDSL includes and type imports. We also see some typedefs we have created for the matrix multiplication DSL. 
We also have some function declarations that we will have in the runtime library for the generated code that we do not want to specialize. Like the function to load matrix from an mtx file. 

Next we have the implementation of the only operator in this DSL - the `mmvp` (Matrix Vector Product) operator. 

This operator takes a matrix (in CSR form), a vector and produces a vector by multiplying them. 
