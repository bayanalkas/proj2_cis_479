# proj2_cis_479
# Probabilistic Robot Localization with Hidden Markov Model (HMM)

## Introduction

This project implements a probabilistic robot localization algorithm using a Hidden Markov Model (HMM). The goal is to estimate the position of a robot in a grid-based environment with sensor measurements and motion updates. The project is implemented in Python and utilizes the NumPy library for efficient matrix operations.

## Usage

1. **Python Installation:** Ensure you have Python installed on your system.

2. **Dependencies:** Install the required dependencies using the following command:
   pip install numpy

# Execution: Run the main Python script to simulate robot localization using the HMM algorithm:
python robot_localization_hmm.py

# Algorithm Details
1. The project implements a Hidden Markov Model (HMM) for robot localization, where the robot estimates its position in a grid environment.

2. The robot can sense its surroundings, and sensor measurements are used to update the estimated probabilities of different locations.

3. The robot can move in different directions, and motion updates are used to predict the robot's new position.

4. The project uses probabilities and conditional probabilities to model sensor measurements and motion updates.

5. The simulation involves a series of actions, including sensing and moving, to demonstrate the HMM-based robot localization algorithm.

# Code Structure
1. robot_localization_hmm.py: The main Python script that implements the HMM-based robot localization algorithm.

2. numpy: The NumPy library is used for efficient matrix operations and calculations.

