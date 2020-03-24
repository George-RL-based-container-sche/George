George
====================
George: Learning to Schedule Long-Running Applications in Shared Container Clusters with Operation Constraints


This repository contains the tensorflow implementation for reinforcement learning based Long Running Application scheduling with operation constraints. 



## Dependencies
1. Python 3.5 or above
2. Tensorflow 1.12.0
3. scipy
4. numpy
5. pandas
6. matplotlib
7. sklearn

## Installation
### Install Requirements
```
git clone https://github.com/George-RL-based-container-sche/George.git
```
### Clone from Github source:
```
pip install -r requirements.txt
```

## Content

Our project includes three parts: 
1. [Real-World LRA cluster](https://github.com/George-RL-based-container-sche/George/tree/master/testbed): Implementation of our seven real-world LRAs that exhibit inter-container interferences.
2. [Testbed](https://github.com/George-RL-based-container-sche/George/tree/master/testbed): George scheduling workflow based on our real-wrold LRA setting.
3. [simulated_env](https://github.com/George-RL-based-container-sche/George/tree/master/simulated_env): George scheduling workflow based on our manually created LRA environment with 20 LRAs.

# Experiment workflow

## Real-World LRA cluster setup and profiler establishment


## George: RL model training for the real-world LRA cluster

1. First check the data collected by [Real-World LRA cluster](https://github.com/George-RL-based-container-sche/George/tree/master/testbed) 
is stored in the folder:

    ```
    $ ls testbed/simulator/datasets/
    ```
    ```
    ***_sample_collected.npz 
    ```

2. Run motivating examples in a 27-node cluster:

   (0) Check the container batches data is stored in the folder or create your own batches:
    ```
    $ ls testbed/data
    ```
    ```
    batch_set_cpo_27node_100.csv batch_set_cpo_27node_100.npz 
    ```

   (1) George: 
    ```
    $ cd testbed
    $ ./RunGeorge.sh 0
    ```
    Output: the training log files including the RPS, constraint violations, time duration .etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
    
    (2) Baseline algorithms: FPO and Subset Search can be executed similarly:
    ```
    $ cd testbed
    $ ./RunFPO.sh 0
    $ ./RunSearchFeasible.sh 0
    ```
    
    (3) Baseline algorithms: Medea is implemented using Matlab, due to its outstanding performance in solving the Integer Linear Programming (ILP) Problem.
        
      Generate the performance-constraints used in Medea:
      
    ```
    $ cd testbed
    $ ./GenerateInterference.sh
    $ ls
    ```
    ```
    interference_applist.csv interference_rpslist.csv
    ```
    Run Medea in the folder:
    ```
    $ cd testbed/Medea
    $ Matlab Medea.m
    ```
    Output: the scheduling decision log files including the allocation matrix, constraint violations, time duration .etc will be store in the folder.

3. Train sub-schedulers in a 27-node cluster: 
    ```
    $ ./TrainSubScheduler.sh
    ```
    Output: the well-trained sub-scheduler models, as well as corresponding log files including the allocation matrix, constraint violations, time duration .etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```

4. High-level model training based on previously well-trained sub-schedulers.

    (0) Check the sub-scheduler models are stored in the folder:
    ```
    $ cd testbed/checkpoint
    $ ls 
    ```
    ```
    cpo_separate_level_0 cpo_separate_level_1 cpo_separate_level_2 ```  
    ```
    Check the container batches data is stored in the `./data/` folder or create your own batches.
    
    (1) High-level training using Transfer Learning, Fine-tune or from scratch:
    ```
    $ ./RunHighLevelTraining.sh 2000
    $ ./RunHighLevelTrainingFineTune.sh 2000
    $ ./RunHighLevelTrainingTS.sh 2000
    ```
    Output: the training log files including the RPS, constraint violations, time duration .etc will be store in the folder:

    
## George: RL model training for the simulated container environment

1. Change the folder to `simulated_env`.

2. Check the datasets stored in `./data/` folder.

3. The 27-node experiments, sub-scheduler training, high-level model training based on sub-schedulers should follow the similar processes above.

Output: the training log files including the RPS, constraint violations, time duration .etc will be store in the folder: `simulated_env/checkpoint/`.


# References
The core of our reinforcement learning algorithm is Constrained Policy Optimization: \[[Paper](https://arxiv.org/abs/1705.10528)\] \[[Code](https://github.com/jachiam/cpo)\]
