import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import *
from process_lidar_data import process_Lidar_Data
from utility import *

# Variable Definitions

MAP                 = {}
log_Odds_Ratio      = np.log(4)
delta_Yaw           = np.zeros(116048)
dt                  = np.zeros(116048)
X                   = np.zeros((116048, 3))
X[0]                = np.array([0,0,0]) # Initializing the first element of X at 0
trajectory          = np.zeros((1,2)) # variable to store the trajectory of the vehicle
theta               = 0 # General variable for angle
encoder_Counter     = 0
lidar_Counter       = 0

# Encoder Parameters

left_Wheel_Dia      = 0.623479 # Meters
right_Wheel_Dia     = 0.622806 # Meters
enc_Resolution      = 4096 # Encoder Resolution

# Particles Parameters

particle_Count      = 40 # Change this to change the count of particles
particle_Weights    = np.zeros((1, particle_Count))
particle_State      = np.zeros((particle_Count, 3))

# Main Code starts from here

# Load Data for all the sensors
timestamp_Lidar, data_Lidar             = initialize_Lidar()
timestamp_Encoder, data_Encoder         = initialize_Encoder()
timestamp_FOG, data_FOG                 = initialize_FOG()

# Initialize MAP
MAP, x_im, y_im, x_range, y_range       = initialize_Map(MAP)

# Sync Encoder and FOG Data
delta_Yaw, dt                           = sync_Encoder_FOG_Data(data_FOG, timestamp_Encoder)

# Initialize Particle Weights
particle_Weights[0, 0:particle_Count]   = 1/particle_Count

# Update Map with the first Lidar Scan
xs0, ys0                                = process_Lidar_Data( data_Lidar[lidar_Counter,:], X[0] )
MAP                                     = update_Map( MAP, xs0, ys0, X[0], log_Odds_Ratio )

# Main Loop
for encoder_Counter in range( len( timestamp_Encoder ) - 1 ):

    # Prediction Step

    left_Wheel_Distance     = ( (data_Encoder[encoder_Counter+1, 0] - data_Encoder[encoder_Counter, 0]) * np.pi * left_Wheel_Dia )/ enc_Resolution
    right_Wheel_Distance    = ( (data_Encoder[encoder_Counter+1, 1] - data_Encoder[encoder_Counter, 1]) * np.pi * right_Wheel_Dia )/ enc_Resolution

    delta_distance          = ( left_Wheel_Distance + right_Wheel_Distance ) / 2

    theta                   = theta + delta_Yaw[encoder_Counter]
    particle_State          = particle_State + np.array( [ delta_distance * np.cos(theta), delta_distance * np.sin(theta), theta ])
    particle_State          = add_Noise( particle_State, particle_Count, dt[encoder_Counter] )

    # Uncomment this line to run without the particles
    # X[encoder_Counter+1]    = X[encoder_Counter] + np.array( [ delta_distance * np.cos(theta), delta_distance * np.sin(theta), theta ])

    # Update Step and Map Update
    if timestamp_Lidar[lidar_Counter] < timestamp_Encoder[encoder_Counter]:

        state_Matched_Particle      = calculate_Map_Correlation(data_Lidar[lidar_Counter,:], particle_Count, particle_State, particle_Weights, MAP, x_im, y_im, x_range, y_range)
        xs0, ys0                    = process_Lidar_Data( data_Lidar[lidar_Counter,:], state_Matched_Particle )
        MAP                         = update_Map( MAP, xs0, ys0, state_Matched_Particle, log_Odds_Ratio )

        # Uncomment these lines to run without the particles
        # xs0, ys0                    = process_Lidar_Data( data_Lidar[lidar_Counter,:], X[encoder_Counter] )
        # MAP                         = update_Map( MAP, xs0, ys0, X[encoder_Counter], log_Odds_Ratio )

        lidar_Counter               += 5 # Change this to change the step size of Update Step

    # Save Trajectory
    X[encoder_Counter]      = state_Matched_Particle
    trajectory              = np.concatenate( ( trajectory, np.array( [[X[encoder_Counter,0], X[encoder_Counter,1]]] ) ), axis = 0 )

    if (lidar_Counter == len(data_Lidar)):
        break

    # Condition to show the map every 10000th iteration
    if encoder_Counter % 10000 == 0:

        x,y = superimpose_Trajectory(trajectory, MAP)
        plot_Map(MAP, x, y)
        print('Encoder Counter:', encoder_Counter)
        print('Lidar Counter:', lidar_Counter)

# Plot the Final Map
x,y = superimpose_Trajectory(trajectory, MAP)
plot_Map(MAP, x, y)
