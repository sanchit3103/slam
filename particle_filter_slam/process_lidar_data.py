import numpy as np
from pr2_utils import *
import matplotlib.pyplot as plt

# Variables for Lidar to Body Transformation

T_L2B                       = np.array([[0.8349, -0.0126869, 1.76416]])
R_L2B                       = np.array([ [0.00130201, 0.796097, 0.605167],
                                         [0.999999, -0.000419027, -0.00160026],
                                         [-0.00102038, 0.605169, -0.796097] ])

# Lidar Parameters

FOV_Lidar                   = 190   # Degrees
start_Angle_Lidar           = -5    # Degrees
end_Angle_Lidar             = 185   # Degrees
resolution_Lidar            = 0.666 # Degrees
max_Range_Lidar             = 80    # meter

# Function Definition

def process_Lidar_Data(data_Lidar, particle_state):

    # Variable Definitions
    Coordinates                 = []
    angle                       = 0
    xs_0                        = np.zeros(286)
    ys_0                        = np.zeros(286)
    
    # Calculate x and y co-ordinates from the Lidar Value
    for i in range( len( data_Lidar ) ):
        angle   = start_Angle_Lidar + ( i * resolution_Lidar )
        temp    = [0,0]

        # Filter Lidar data with range less than 2m and greater than 75m
        if data_Lidar[i] > 2 and data_Lidar[i] < 75:
            temp[0] = data_Lidar[i] * ( np.cos( np.deg2rad( angle ) ) )
            temp[1] = data_Lidar[i] * ( np.sin( np.deg2rad( angle ) ) )
            Coordinates.append(temp)

    # Conversion from Lidar to World frame

    # Formation of pose matrix for Sensor to Body Frame transformation
    pose_Matrix_L2B = np.concatenate( ( R_L2B, T_L2B.T ), axis = 1 )
    pose_Matrix_L2B = np.concatenate( ( pose_Matrix_L2B, np.array( [[0,0,0,1]] ) ), axis = 0 )

    # Formation of pose matrix for Body to World Frame transformation
    theta           = particle_state[2]
    R_B2W           = np.array([ [np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1] ])

    pose_Matrix_B2W = np.concatenate( ( R_B2W, np.array( [[particle_state[0], particle_state[1], 0]] ).T ), axis = 1 )
    pose_Matrix_B2W = np.concatenate( ( pose_Matrix_B2W, np.array( [[0,0,0,1]] ) ), axis = 0 )

    # Formation of final pose matrix for Sensor to World Frame transformation
    pose_Matrix     = np.dot(pose_Matrix_B2W, pose_Matrix_L2B)

    for i in range(len(Coordinates)):
        temp = [Coordinates[i][0], Coordinates[i][1], 0, 1 ]
        temp = np.dot(pose_Matrix, temp)
        #if temp[2] < 10:
        xs_0[i] = temp[0]
        ys_0[i] = temp[1]

    xs_0 = xs_0[xs_0 != 0]
    ys_0 = ys_0[ys_0 != 0]

    return xs_0, ys_0
