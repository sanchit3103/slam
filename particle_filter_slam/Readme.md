# Implementation of Particle Filter SLAM using Odometry, 2-D LiDAR Scans and Stereo Camera Measurements 

<p align="justify">
The project focuses on the methodology to implement Particle Filter SLAM using odometry data from encoders and Fiber Optic Gyroscope (FOG), 2-D LiDAR scans and stereo camera measurements. Particle Filter SLAM uses a set of particles to represent the vehicle’s state. The particle representing the most accurate state of vehicle according to the map of environment is considered for localization. This cycle continues for all data points of sensors and the path covered by the vehicle. Mapping is performed using log-odds 2-D occupancy grid map. It is updated for every available scan data from LiDAR. The results of the path followed by the robot in dead reckoning state and after implementation of particle filter are presented in the report.
</p>

## Project Report
[Sanchit Gupta, 'Implementation of Particle Filter SLAM using Odometry, 2-D LiDAR Scans and Stereo Camera Measurements', ECE 276B, Course Project, UCSD](https://github.com/sanchit3103/slam/blob/main/particle_filter_slam/Report.pdf)

## Project Implementation

#### a) Dead Reckoning Trajectory of the vehicle obtained from Prediction-only Particle Filter

<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/209096203-644aa952-5c44-4da4-90ba-b357cbc96788.png" height="300"/>
  
</p>

#### b) Occupancy Grid Map at three different stages of the trajectory

<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/209097004-d0bbbc21-d8c0-467d-94fa-816b2ea48e71.jpg" height="300"/>, &nbsp;&nbsp; <img src = "https://user-images.githubusercontent.com/4907348/209097081-3bd9c06f-210a-4d8c-8ad1-3954eae12069.jpg" height="300"/>, &nbsp;&nbsp; <img src = "https://user-images.githubusercontent.com/4907348/209097136-081fbce2-9a26-4f3b-9834-f3b757ec28e5.jpg" height="300"/> 
  
</p>

## Details to run the code:

* <b> main.py: </b> Main file which should be run to execute the project. This file contains the main loop executing Prediction Step, Update step and Map update of Particle Filter SLAM.
* <b> utility.py: </b> Contains all the utility functions required for the project.
* <b> process_lidar_data.py: </b> Contains the function definition to transform LiDAR scan points into x and y co-ordinates, their conversion from sensor to body frame and subsequently to world frame.
* <b> pr2_utils.py: </b> Contains utility functions for map update.

