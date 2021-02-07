# Out of Memory

This page illustrates how to exceed the memory alloction and the resulting OOM error message on both the CPU and GPU.

```python
import numpy as np
x = np.random.randn(200_000_000).astype(np.float64)
```

## CPU

```python
import numpy as np
from time import sleep

N = 500000000
x = np.random.randn(N)
sleep(10)
y = np.random.randn(N)
sleep(10)
```

Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=mem-exceed    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem=5G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)

module purge
module load anaconda3

srun python mem.py
```

Here is the resulting error message:

```
srun: error: tiger-i23g11: task 0: Out Of Memory
srun: Terminating job step 3955284.0
slurmstepd: error: Detected 1 oom-kill event(s) in step 3955284.0 cgroup. Some of your processes may have been killed by the cgroup out-of-memory handler.
```

## GPU

```
$ ssh tigergpu.princeton.edu
$ module load anaconda3
$ conda create --prefix /scratch/gpfs/$USER/torch-env pytorch cudatoolkit --channel pytorch
```

Python script:

```python
import torch
from time import sleep

N = 1200000000

x = torch.randn(N, dtype=torch.float64, device=torch.device('cuda:0'))
sleep(10)
y = torch.randn(N, dtype=torch.float64, device=torch.device('cuda:0'))
sleep(10)
```

Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=mem-exceed    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem=25G                # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate /scratch/gpfs/$USER/torch-env

srun python mem.py
```

Here is the resulting error message:

```
Traceback (most recent call last):
  File "mem.py", line 8, in <module>
    y = torch.randn(N, dtype=torch.float64, device=torch.device('cuda:0'))
RuntimeError: CUDA out of memory. Tried to allocate 8.94 GiB (GPU 0; 15.90 GiB total capacity; 8.94 GiB already allocated; 6.34 GiB free; 0 bytes cached)
srun: error: tiger-i23g11: task 0: Exited with exit code 1
srun: Terminating job step 3955266.0
```


## C++

```c++
// g++ -std=c++11 mem_test.cpp

#include <iostream>
#include <random>
#include <unistd.h>

int main(int argc, char* argv[]) {
  std::cerr << sizeof(double) << " bytes per double" << std::endl;
  long N = 400000000;
  double* chuck = new double[N]; // 8 Bytes x 4e8 = 3.2GB
  //double arr[N];
  std::cerr << "Storing random numbers ..." << std::endl;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < N; i++)
    chuck[i] = distribution(generator);
  std::cerr << "Done generating random numbers." << std::endl;
  std::cerr << sizeof(chuck) << std::endl;

  std::cerr << "Sleeping for 10 seconds ..." << std::endl;
  unsigned int microseconds = 10000000;
  usleep(microseconds);
  std::cerr << "Done sleeping." << std::endl;

  delete[] chuck;
  return 0;
}
```
