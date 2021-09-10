This procedure and document is attached as an appendix to the dissertation
‘Reinforcement learning and foraging optimisation for controlling underactuated vibrating robot’

The following is the python code used for this study and the CoppeliaSim file. The sim.py and simConst.py files in the CoppeliaSim_RL-main(bristle) folder are necessary to connect to the CoppeliaSim environment.
First you need to download and install CoppeliaSim 4.1.0 as the training environment.
If the model needs to be trained, the necessary python libraries are.
	Numpy
	Pytorch
	Matplotlib
	tensorboardX
	rlschool
First you need to open the bristle1.ttt file and the train.py file, and run the train.py file directly to start training.
The network for the algorithm is written in the model.py file for the main program to call and for easy modification.


