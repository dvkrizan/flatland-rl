# Reinforcement Learning Applied to Transportation
###### Capstone students: Claire-Isabelle Carlier and David Krizan

## Overview and Deliverables
Our Capstone Project goal was to learn about Reinforcement Learning and apply it to a real-world problem. 
We opted to work with the Flatland environment. Flatland is an open-source toolkit for developing and comparing Multi-Agent Reinforcement Learning algorithms applied to the [vehicle rescheduling problem (VRSP)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.5205&rep=rep1&type=pdf). VRSP seeks to address the situation where a vehicle breaks down or is delayed during operation in a public transportation system, where other vehicles need to be rescheduled to minimize the impact of the issue on the network.
The Flatland environment has been developed by the Swiss Federal Railway (SBB) and Deutsche Bahnn, in collaboration with AICrowd, a platform for AI Competition (see credits at the bottom). Several competitions were lead using this environment, including the recent NeurIPS 2020 Challenge. 

As a lot of code had already been developed for this environment (and more code was released over the summer), we focused on understanding the concepts, debugging the environment, optimization the learning process, creating visualizations of our results and playing with hyperparameters to improve existing code. 
Our deliverables include:
- This git repository to run the code (please see below for instructions on how to run the code and generate GIF clips
- A [video introduction to core concepts of Reinforcement Learning applied to the Flatland environment](link) that we strongly encourage you to watch before reading the blog and playing with the code.
- A [blog post on Proximal Policy Optimization](link), an on-policy technique to solve the Flatland challenge, that includes a code walk-through.


## How to run the code
### Set up the environment
To be able to run the code, you should follow the following steps (we assume you use conda to host your environments):
1. Create an environment to host your Flatland experiments. Note that you will need to use python 3.6 for compatibility reasons:
`conda create python=3.6 --name flatland-rl`
2. Once your environment has been created, you need to activate it:
`conda activate flatland-rl`
3. Now that your environment has been created, install the flatland-rl python package via conda:
`pip install flatland-rl`
4. You will need additional packages to be able to run the code. Here are the libraries you should be installing:
`pip install argparse psutil tensorboard`
5. If you have a NVIDIA GPU on your computer, install a [recent version of the CUDA toolkit (9 or above)](https://developer.nvidia.com/cuda-toolkit) on your computer, then run the following command to tie it to pytorch (do not install pytorch prior to this):
`conda install pytorch torchvision cudatoolkit=<your version of cuda> -c pytorch`
Otherwise, you will need to pip install pytorch. 
6. To generate nice videos and GIFs of your runs, you will need FFMPEG:
`conda install ffmpeg` (do not pip install if you are on Windows!)
7. Test the demo to make sure everything was installed properly:
`flatland-demo`
This should pop up a window with little trains moving around. 
Note that you can install a full starter-toolkit from the Flatland github repository but it contains a lot of files to prepare for competition submission which we don't think is useful for the purpose of this project.


### Run the single agent or multi-agent files
To run the single agent file, make sure your `flatland-rl` environment is activated and run the single_agent.py as follow:
`python single_agent_training.py -n <number of episode you wish to train for>`

To run the multi_agent.py file, you need to input an additional set of parameters. We recommend:
`python multi_agent_training_ppo.py -n 1000 -t 1 -e 1 --n_evaluation_episodes 20 --eps_start 0.8 --batch_size 20 --use_gpu True --n_agent_fixed`
This will run 1000 episodes with the Test 1 configuration (5 agents, 25x25 grid, malfunction rate of 1/50), use 20 episodes for evaluation and an maximum epsilon threshold of 0.8. 
If will also run on GPU if you have one available. Number of agent will stay the same across each episode.


### Generate videos and GIFs
We have already included code to capture images at different time step (currently set up to capture them every 100th episode). 
To generate a mp4 out of the images, navigate to an episode's folder and run the code below (example is with episode 0):
`ffmpeg -y -framerate 12 -i flatland_frame_%04d.png -hide_banner -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" episode_0.mp4`
You then need to generate a palette out of the video to generate beautiful gifs:
`ffmpeg  -i episode_0.mp4 -filter_complex "[0:v] palettegen" palette.png`
and finaly you can generate the gif
`ffmpeg -i episode_0.mp4 -i palette.png -filter_complex "[0:v][1:v] paletteuse" episode_0.gif`






🚂 The code is this repository is based on the official starter kit - NeurIPS 2020 Flatland Challenge and has been modified for the purpose of our Capstone project.
---

NeurIPS Credits
---

* Florian Laurent <florian@aicrowd.com>
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Sharada Mohanty <mohanty@aicrowd.com>
* Christian Baumberger <christian.baumberger@sbb.ch>
* Guillaume Mollard <guillaume.mollard2@gmail.com>

Main links
---

* [Flatland documentation](https://flatland.aicrowd.com/)
* [Flatland Challenge](https://www.aicrowd.com/challenges/flatland)

