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
The purpose of our paper is to show that it is easy to write DSLs with staging and BuilDSL. To demonstrate that we will write a simple DSL for matrix multiplication that generates GPU code. We will use static variable to implement an analysis and specialization for automatically moving buffers between HOST and DEVICE. 

We have provided the skeleton code for this section in `mm-dsl-skeleton.cpp`. You can start by copying this into a new file with the following command in the top level directory of this repository - 

```
cp mm-dsl-skeleton.cpp mm-dsl.cpp
```

You can now run the DSL and the application to generate some code. At any point to compile and execute the DSL run - 

```
bash run_dsl.sh
```

If you have successfully compiled the dependencies and copied the skeleton file, you should see some generated C++ code. We will inspect the source file and the generated code. To understand how the DSL is implemented and to add more features open the `mm-dsl.cpp` in your favorite editor. We have `vim` and `nano` installed on our provided system. If you wish to use other editors, you can `scp` the file, modify it and upload it again. 

Upon opening the file, you can see the code is divided into many parts as follows - 

  - Runtime Functions Declaration: This section declares a bunch of functions our DSL can assume are available in the runtime. This includes functions like malloc, memcpy and other CUDA functions. We will not provide the implementation of these to the DSL because they don't need specialization. 
  - Dual Array Type Declaration: In this section we define a templated type `dual_array` that can store an array of values on both CPUs and GPUs and move them around as required. This type has two "dynamic" pointers, one pointing to a buffer on the host and one pointing to a buffer on the GPU. It also has a static variable to determine where the actual values are currently located. This variable will change through the execution of the program as we move the buffers around. We also have a global variable to determine where the code is currently executing (`current_context`). This helps us generate different implementations for sections running on HOST and DEVICE. 
  - Matrix Vector Product Implementation: This section actually defines the Matrix vector product operation - `y = M * x` where x and y are vectors and M is a square matrix. 
  - Test App Written With DSL: This section has a simple application written using the DSL defined above. This app uses the types and functions we defined in the DSL. 
  - Main Driver: Finally the section that invokes BuilDSL to generate code for the DSL based on inputs. 

You can notice that there are a few TODOs in the code left for you to complete. Let us fill them one by one and see how the generated code changes. 

#### TODO#1
Find the TODO#1 in the code. This is in the member function of our dual array type that returns the value at a particular index. You can see this function has 4 specializations based on where the particular buffer is currently located and where the currrent code is running. You can see that the implementation for when the buffer is located where the code is currently executing is simple and just returns the value from the corresponding buffer. 

You are supposed to fill in the implementation when the buffer is located on the DEVICE, but the code is running on the HOST. You can insert the following implementation - 

```
dyn_var<T> temp;
runtime::to_host(&temp, &(array->device_buffer[index]), (int)sizeof(T));
return temp;
```

What this implementation does is just moves the exact index to the HOST that we need to read and returns it. Such a specialization allows to make adjustments to the array from the host without copying the whole array back and forth. The second function `operator =` also has a similar specialization which is already filled it. We use the the static variable `current` and `current_context` to determine which specialization to invoke. You can run the DSL again with the command above, but nothing will change because haven't invoked this specialization yet. 

#### TODO#2
Find the TODO#2 in the code. This is in the implementation of our `mmvp` operator. We have an extra parameter to the function that statically tells it where to execute the logic. We have already filled in the logic for executing on the CPU serially. You have to fill in the code for execution on the GPU. 

You can insert the following implementation - 

```
current_context = DEVICE;
dyn_var<int> num_cta = (n + 511) / 512;

builder::annotate(CUDA_KERNEL);
for (dyn_var<int> cta = 0; cta < num_cta; cta = cta + 1) {
        for(dyn_var<int> tid = 0; tid < 512; tid = tid + 1) {
                dyn_var<int> r = cta * 512 + tid;
                if (r < n) {
                        y[r] = 0.0f;
                        for (dyn_var<int> c = 0; c < n; c = c + 1) {
                                y[r] = y[r].get() + M[r * n + c].get() * x[c].get();
                        }
                }
        }
}
current_context = HOST;
```

This code does two things, first it sets the `current_context` to DEVICE. This tells all the specialization that the code is now running on the GPU. Next we use BuilDSL's CUDA annotation "CUDA_KERNEL" explained in Figure 5 of the paper to tell the framework that this loop nest should be mapped to a GPU. The inner most loop that iterates over the column is exactly the same, while individual rows are mapped to separate threads. 

We can also navigate to the `test` function where there are two calls to the `mmvp` functions. We can change the second call (even iteration) to run on the GPU. Finally the function should look like this - 

```
if (iter % 2 == 0)
        mmvp(n, M, x, y, HOST);
else
        mmvp(n, M, y, x, DEVICE);
```

At this point let us try running the code with the command `bash run_dsl.sh`. 
You will notice that the execution fails with an error -

> "Cannot move host array to device from device. Please insert moves at appropriate places"

This is because our static analysis has figured out that the buffer currently resides on the HOST and is required on the DEVICE. There is no specialization possible for this because code executing on the GPU has no API functions to pull data from the host. 

#### TODO#3
To fix this we will have to manually move the whole array to the GPU when we need it. Find TODO#3 in the code and insert the following 3 lines - 

```
M.move_to_device();
x.move_to_device();
y.move_to_device();
```

This moves all the 3 arrays to the GPU before invoking the GPU kernel. Now we can run the DSL and you will notice that the generated code has 2 calls to CUDA kernels and the first one has 3 calls to `to_device`, a function that copies arrays from the host to the GPU. This code is now correct and should produce the right result. 

But we can see and interesting difference for the next iteration that runs on the host (`iter == 2`). Because now the buffers have been moved to the GPU, the host side code whenever it needs to read or write these values, moves individual indices back and forth (this is result of the specialization we wrote in TODO#1). This is correct but highly ineffecient. In our `mmvp` function we know that the whole array is going to be requied on the host, so we can move it back completely before we start execution. 

#### TODO#4
Find the TODO#4 in the code. This is in the host side implementation of the `mmvp` function. We can add the following code there - 

```
M.move_to_host();
x.move_to_host();
y.move_to_host();
```

These functions will move the arrays to the host upfront. Now the calls to the `get()` and `=` operators don't actually have to move the values one by one. Run the dsl again to see the final code generated showing 4 matrix multiplication operations 2 on the host and 2 on the GPU. 

We have utilized the CUDA extraction pass from the framework and our methodology of using static variables to track properties about runtime values. 

We have also provided the completed implementation in `mm-dsl-finished.cpp` if you need a reference at any point. 

With this we have concluded the artificat evaluation of our paper. We have shown - 

  - The graph DSL we implemented in BuilDSL generates code at par with the state of the art graph DSL GraphIt
  - We demonstrated how a simple DSL for matrix multiplication with GPU and CPU code and automatically managed data transfers can be implemented in few lines of code (without runtime conditions and overheads)

