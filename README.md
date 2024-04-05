# Reinforcement Learning for supply chain management

This project focuses on applying reinforcement learning (RL) techniques to optimize supply chain management, specifically in the context of a hydrogen production plant supplying various resale sites. The problem involves considerations such as transport costs, production costs, and stock capacity. Two RL algorithms, Proximal Policy Optimization (PPO) and Double Deep Q-Network (D-DQN), are implemented and compared against random agent and human performance benchmarks. Results show that D-DQN outperforms other methods, prompting further analysis of its robustness through experiments involving noisy demand forecasts and the addition of new resale sites. The conclusion drawn from the study highlights the efficacy of RL-based approaches in addressing complex supply chain management problems. However, it also acknowledges limitations such as the need for retraining models when parameters change significantly or when new sites are added, indicating avenues for future research and improvement. This work contributes to understanding the applicability and challenges of RL in real-world supply chain optimization scenarios.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The objective of this project is to automate the management of a production plant in a very simple case in order to evaluate the potential of Reinforcement Learning in this field. In concrete terms, we have modeled a hydrogen production company that supplies a certain quantity of gas to different sites every day. Our goal is as follows: based on consumption forecasts, transportation costs, production, storage, selling price... how to optimize the management of the hydrogen production of the plant as well as the supply to different sites in order to maximize profit over a finite time horizon. In fact, the performance of the different agents will be measured in k\$ and will be the profit that such an agent is able to achieve.
To this end, we have implemented our own environment capable of modeling the problem (no environment of this kind exists at the moment) as well as intelligent agents capable of addressing the problem. Furthermore, we have studied the influence of different parameters of the algorithms and the environment on the performance of our agents.

## Implementation

In this section, please read main.py file (+ doc on top) and main_experiments.ipynb for details on our implementation of experiments (+ see report).

## Installation

Prerequisites/dependencies:

asttokens==2.4.1
ca-certificates==2023.12.12
cloudpickle==3.0.0
colorama==0.4.6
comm==0.2.1
contourpy==1.2.0
cycler==0.12.1
debugpy==1.8.1
decorator==5.1.1
exceptiongroup==1.2.0
executing==2.0.1
farama-notifications==0.0.4
filelock==3.13.1
fonttools==4.49.0
fsspec==2024.2.0
gym==0.26.2
gym-notices==0.0.8
gymnasium==0.29.0
importlib-metadata==7.0.1
importlib-resources==6.1.2
ipykernel==6.29.3
ipython==8.18.1
jedi==0.19.1
jinja2==3.1.3
jupyter-client==8.6.0
jupyter-core==5.7.1
kiwisolver==1.4.5
markupsafe==2.1.5
matplotlib==3.8.3
matplotlib-inline==0.1.6
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.2.1
numpy==1.26.4
openssl==3.0.13
packaging==23.2
parso==0.8.3
pillow==10.2.0
pip==23.3.1
platformdirs==4.2.0
prompt-toolkit==3.0.43
psutil==5.9.8
pure-eval==0.2.2
pygments==2.17.2
pyparsing==3.1.1
python==3.9.18
python-dateutil==2.8.2
pywin32==306
pyzmq==25.1.2
scipy==1.12.0
setuptools==68.2.2
six==1.16.0
sqlite==3.41.2
stack-data==0.6.3
sympy==1.12
text-flappy-bird-gym==0.1.1
torch==2.2.1
torchaudio==2.2.1
torchvision==0.17.1
tornado==6.4
tqdm==4.66.2
traitlets==5.14.1
typing-extensions==4.10.0
tzdata==2024a
vc==14.2
vs2015_runtime==14.27.29016
wcwidth==0.2.13
wheel==0.41.2
zipp==3.17.0


## Usage

1. Provide examples of how to use your project.
2. Include any necessary configurations or settings.
3. Explain any command-line options or parameters.

## Contribution

This project is a collaboration between Mehdi EL KANSOULI, Victor GIROU and Samuel PARIENTE in CentraleSupelec. 


