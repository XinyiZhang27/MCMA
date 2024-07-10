# MCMA
A Mobility-Aware Cooperative Multi-Agent (MCMA) deep reinforcement learning approach for the task migration-assisted multi-edge computation offloading problem in vehicular networks.

Four scenarios are simulated with Simulation of Urban MObility (SUMO) (https://sumo.dlr.de/docs/index.html), including $\mathrm{Grid}_{3\times 3}$, $\mathrm{Net}_4$, Pasubio, and A.Costa.  
Download the datasets and build the directory yourself. The simulated datasets are available at https://drive.google.com/drive/folders/1oXeOSBLP-BjTtyfzmWor61gPqSVROk26?usp=sharing.

# File Structure
* /algorithms: Implementation of MAPPO and MADDPG algorithms.
* /envs: Task migration-assisted multi-edge computation offloading environment.
* /myutils: Implementation of MAPPO and MADDPG buffers.
* /runner/separated: Codes for MCMA training and test.
* /st_prediction: Informer-based multi-step vehicular trajectory prediction module.
* /sumo: Four simulated datasets.
* main.py: The main function.
* config.py: The configuration file.

# Model Training
1. Download the simulated datasets and put folder "sumo" under folder "MCMA".
2. Under directory /st_prediction:  
   1) Run data_processing.py and generate processed data in folder "trajectory_data".  
   2) Pre-train vehicular trajectory prediction models with main_informer.py. The pre-trained prediction models are saved in folder "checkpoints".
4. In file main.py:  
   1) Set "--stage" to "train" and run main.py to train MCMA.  
   2) Note that "--simulation_scenario", "--time_range", and "--num_edge" need to be modified according to different scenarios.

# Model Test
1. file main.py:  
   1) Set "--stage" to "train" and run main.py to test MCMA. Set the test episodes with "start_epi" and "end_epi".  
   2) Note that "--model_dir1" and "--model_dir2" should be filled with the paths of the trained MAPPO and MADDPG models, respectively.

# Requirements
python 3.8  
torch 1.13.1+cu116  
tensorboard 2.14.0
