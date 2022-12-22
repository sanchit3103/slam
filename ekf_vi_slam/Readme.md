# Visual Inertial SLAM using Extended Kalman Filter

<p align="justify">
This project focuses on the methodology to implement Visual-Inertial Simultaneous Localization and Mapping (SLAM) using Extended Kalman Filter (EKF) and the synchronized measurements provided from an inertial measurement unit (IMU) and a stereo camera. EKF prediction and update steps are implemented in this project to obtain the dead reckoning trajectory of the robot and to estimate the positions of landmarks observed during the path. The complete Visual-Inertial SLAM algorithm is then implemented combining the IMU prediction step and landmark update step to obtain updated trajectory of the robot. The results of the path followed by the robot in dead reckoning state, updated positions of the landmarks and updated robot’s trajectory after Visual-Inertial SLAM for different number of features are presented in the report.
</p>

## Project Report
[Sanchit Gupta, 'Visual Inertial SLAM using Extended Kalman Filter', ECE 276A, Course Project, UCSD](https://github.com/sanchit3103/slam/blob/main/ekf_vi_slam/Report.pdf)

## Project Implementation

#### a) Dead Reckoning trajectories of the vehicle obtained from Prediction Step
<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/209108478-bd1ee4d8-b278-4adf-80d1-039801a95447.png" height="350"/>
  <img src = "https://user-images.githubusercontent.com/4907348/209108497-ef90f3a7-6e0c-49db-a6a1-bf246ae305fd.png" height="350"/>
  
</p>

#### b) Plots after landmark update steps for both the datasets with number of features = 1000
<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/209109157-f16aec8c-56e8-4dac-8744-d237a0db90f9.png" height="220"/>
  <img src = "https://user-images.githubusercontent.com/4907348/209109174-b5e7daab-68ea-4cf9-9074-22c76d52c5ed.png" height="220"/>
  
</p>

#### c) Plot after executing the complete Visual-Inertial SLAM algorithm
<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/209109664-8dc784bc-0ae0-49e9-88fd-aa4373021560.png" height="220"/>
  
</p>

## Details to run the code:

* <b> main.py: </b> Main file which should be run to execute the project. This file contains the main loop executing EKF Prediction Step, EKF Update step for the landmarks and combined Visual SLAM algorithm.
* <b> utility.py: </b> Contains all the utility functions required for the project.
* <b> prediction_step.py: </b> Contains the function definition to execute IMU localization via EKF Prediction.
* <b> pr3_utils.py: </b> Contains utility functions for trajectory update.

