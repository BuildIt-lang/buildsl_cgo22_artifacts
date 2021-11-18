This repository documemts the evaluation procedure for the artifacts of our paper "GraphIt to CUDA compiler in 2021 LOC: A case for high-performance DSL implementation via staging with BuilDSL". The artifacts are divided into 2 sections - 
1. Reproduce the results from Figure 11 in the paper that compares the performance of code generated from BuilDSL with the code generated from the state of the art DSL GraphIt. 
2. (Optional) Use the Generalized DSL framework presented in the paper to implement a very simple matrix multiplication DSL. 

The requirements (software and hardware) for each sections are - 

Section 1:
	Linux system preferable Ubuntu 18.04 or higher
	GCC 7 or higher (or clang equivalent)
	CUDA 10.0 or higher
	python3
	cmake 3.13 or higher
	NVIDIA GPU (preferable Volta generation NVIDIA V-100 32 GB)
	
Section 2:
	Linux system preferable Ubuntu 18.04 or higher
	GCC 7 or higher (or clang equivalent)

All the requirements for both the sections are available on the system provided for artifact evaluation. Please refer to the artifact evaluation pdf for instructions on how to connect to our servers. 

You can run these experiments on your own server with any NVIDIA GPU. The results presented in the paper have been collected on a system with an NVIDIA V-100. The results might not match exactly if you use a different GPU but should be similar. 

To start, first clone this repository using the following command - 

```
git clone --recursive <insert git url here>
```

If you have already cloned the repository without the recursive flag, you can run the following commands inside the cloned directory to fetch all the submodules - 

```
git submodules init
git submodules update
```

Now navigate to the main repostitory and continue the rest of the steps. 

## Step 1: Build all dependencies
We will first build all the dependencies which have been placed in the repository as submodule. This includes the GraphIt GPU compiler repository, the BuilDSL repository and the BuildIt repository inside BuilDSL that it depends on. 

To build all the dependencies, simply run the following command from the top level directory of this repository - 

```
bash build_all_deps.sh
```

This command should take maximum of 10 mins to build all the dependencies. If any of the particular submodules fails to build, it will be reported and you can try to build it again. 


## Step 2: Fetch the datasets
The evaluation for Figure 11 requires obtaining 9 graph dataset files. **If you are running this evaluation on your own server, you will have to download the dataset. This might take some time depending on your network connection**. If you are running the evaluation on own provided system, we can directly softlink the files into the dataset directory. This will be quick. 

Navigate to the dataset directory by running the command from the top level directory of this repository - 

```
cd dataset
```

If you are running on your server, run the command - 

```
make dataset
```

if you are running on our server, run the command - 

```
make local
```

If everything runs okay, you should see 9 `.mtx` files in the dataset directory. You can verify this by running the `ls` command. Now navigate back to the top level directory. 


## Step 3: Section 1: Reproduce Figure 11
The purpose of this evaluation is to run all the 5 applications on all the 9 datasets with both GraphIt and BuilDSL and compare their execution times. Before we run the actual script to run all the experiments, we will find a free GPU on the system to run our experiments. It is essential that the GPU we run our experiments on doesn't have any other processes running on it as it could interfere with our execution times. The server we have provided for this evaluation has 8 NVIDIA V-100 GPUs that are all equivalent. Some of them might be currently in use. So please follow the steps carefully to find a free GPU. 

Start by running the command - 

```
nvidia-smi
```

This command will list all the GPUs installed on the server, their IDs (indexed from 0) and thr processes running on each of them. From this list, find a GPU index that does not have any processes running on it. The processes are listed at the bottom of the output of the above command. If no processes are listed, all the GPUs are free. Note down this index to be provided in the next step. 


Now run the command - 

```
python3 gen_fig11.py
```

Immediately this command will prompt a few configuration options (with default options in []). If you are running on our server, all the default values would be appropriate. You can select the default by simply pressing enter. 

One of the prompts will ask the GPU index to use. Just type the index you had noted above. The default here will be zero and can be selected by just pressing enter as before. 

If you are running on your own servers, the only options you might have to specify are path to the CXX compiler and the path to the CUDA SDK. Again, if you have installed these dependencies with the package manager, the default options should be fine. 

The script will then start running all the evaluations. This step will take 45mins - 1 hr depending on the system. The scripts runs all the experiments above 10 times and takes the min execution time in each case to avoid any regressions arising from other processes in the system. 
Once the script completes, it will print a table with the execution times for the 5 applications and 9 datasets for both GraphIt and BuilDSL. The numbers from both the frameworks will be in the same cell separated by a `/`. (GraphIt/BuilDSL). The main aim of this experiment is to show that BuilDSL generates comparative code to the state of the art graph DSL GraphIt. 

You can also view this table later by running the following command from the top-level directory of this repository - 

```
cat outputs/fig11.txt
```

At this time, you can also browse the code generated from BuilDSL in the outputs directory - 

```
less outputs/bfs_power_buildsl.cu
less outputs/bfs_road_buildsl.cu
```

These files show the code generated for two different schedules for the Breadth First Search application. These code snippets are in the supplementary section of our original paper. 

## (Optional) Step 4: Section 2: Write a simple DSL with BuilDSL
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

This operator takes a matrix (in COO form), a vector and produces a vector by multiplying them. 

