# Pedestrian Trajectory Prediction - Master Thesis

This repository is part of my master thesis, focusing on pedestrian trajectory prediction. The project aims to compare classical approaches such as Bayesian filters and particle filters with modern neural network-based approaches like the Human Scene Transformer (HST). The key objective is to determine if neural networks provide a tangible advantage for simple pedestrian trajectory prediction problems like single trajectories in open spaces.

IT IS STILL IN PROGESS AND NOT FINISHED OR USABLE !!!

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Implemented Approaches](#implemented-approaches)
  - [Bayesian and Particle Filters](#bayesian-and-particle-filters)
  - [Human Scene Transformer](#human-scene-transformer)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Human Scene Transformer](#training-the-human-scene-transformer)
  - [Running Bayesian and Particle Filters](#running-bayesian-and-particle-filters)
  - [Comparative Analysis](#comparative-analysis)
- [Results and Discussion](#results-and-discussion)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Pedestrian trajectory prediction is a crucial task in various applications, including autonomous driving, robotics, and crowd management. Classical approaches like Bayesian filters have been widely used for this purpose due to their robustness and interpretability. However, with the advent of deep learning, methods such as the Human Scene Transformer have shown promise in handling complex interactions in dynamic environments.

The primary goal of the thesis is to predict the future positions of pedestrians given the visual and motion information from a mobile robot. The use case for the trajectory prediction is an environment with no or only few obstacles, further called an open space scenario, as well as only few pedestrians. Most work done in pedestrian trajectory prediction is intended for scenarios with either few pedestrians but a lot of context and bias, e.g. in traffic, or with less context but crowds of people and the focus on social interaction, e.g. in surveillance views. Therefore, research for non-crowded open spaces from an ego-perspective is rare.

The secondary goal of the thesis is to validate the hypothesis that using a state-of-the-art deep learning method in the open space scenario is more useful than using an algorithmic approach like Bayesian Filter combined with a model that describes human motion.
The hypothesis is based on the assumption that it is not sufficient to only model a pedestrian as a 2D point mass. Instead, a focus on the individual body pose and head orientation might be necessary in an open space scenario. 

## Repository Structure

The repository is nto fully organized until now, but will come soon ...


## Implemented Approaches

### Bayesian and Particle Filters

The repository includes several Bayesian filters commonly used for pedestrian trajectory prediction:

- **Kalman Filter**: A linear quadratic estimation algorithm used for filtering and predicting the state of a dynamic system.
- **Interacting Multiple Model (IMM) Filter**: A probabilistic filter that combines multiple Kalman filters for tracking objects with different motion models.
- **Particle Filter**: A non-parametric filter that approximates the posterior distribution of the state using a set of weighted particles. It is particularly useful for non-linear and non-Gaussian tracking problems.

### Human Scene Transformer

The Human Scene Transformer (HST) is a neural network model designed to predict pedestrian trajectories by modeling the social interactions in dynamic environments. This implementation is adapted to handle single pedestrian trajectories and includes:

- **Transformer Architecture**: The core of the HST model, which uses self-attention mechanisms to capture spatial-temporal dependencies.
- **Training and Evaluation Scripts**: Scripts for training the model on pedestrian trajectory data and evaluating its performance.

## Installation

To set up the environment, clone the repository and install the required dependencies:

git clone https://github.com/your-username/pedestrian-trajectory-prediction.git
cd pedestrian-trajectory-prediction
pip install -r requirements.txt

## Usage
ToDo

Based on https://github.com/google-research/human-scene-transformer/tree/main and
```
@article{salzmann2023hst,
  title={Robots That Can See: Leveraging Human Pose for Trajectory Prediction},
  author={Salzmann, Tim and Chiang, Lewis and Ryll, Markus and Sadigh, Dorsa and Parada, Carolina and Bewley, Alex}
  journal={IEEE Robotics and Automation Letters},
  title={Robots That Can See: Leveraging Human Pose for Trajectory Prediction},
  year={2023}, volume={8}, number={11}, pages={7090-7097},
  doi={10.1109/LRA.2023.3312035}
}
