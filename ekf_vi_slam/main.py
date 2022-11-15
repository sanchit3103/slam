import numpy as np
from pr3_utils import *
from utility import *
from prediction_step import *
from scipy.linalg import expm

# Load the measurements
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

# Variable Definitions
best_Features_Count = 200
features_WF         = np.zeros((features.shape))
mu_Features         = np.zeros((4, best_Features_Count, features.shape[2]))
mu_Features_Flatten = np.zeros((3*best_Features_Count, 1))
Ks                  = formulateKs(K,b)
cam_T_imu           = np.linalg.inv(imu_T_cam)
P_Transpose         = np.concatenate( (np.eye(3), np.array([[0,0,0]])), axis = 0)
sigma_t             = np.eye(3*best_Features_Count)*0.01 # 3m x 3m
feature_frequency   = np.zeros((best_Features_Count))


if __name__ == '__main__':

	# (a) IMU Localization via EKF Prediction

	# IMU Pose Prediction Step
	trajectory, sigma_Prediction = prediction_Step( t, linear_velocity, angular_velocity ) # 4x4xtime
	print('Trajectory completed')

	# Compute World frame coordinates of all features across all timestamps
	for i in range(features.shape[2]):
	    features_WF[:,:,i] = compute_WF_Coord(features[:,:,i], Ks, imu_T_cam, trajectory[:,:,i])

	print('WF completed')

	# Get the best features basis their frequency of appearing
	best_Features_WF, filtered_Features     = best_Features(features_WF, features, best_Features_Count) # best_Features_WF is the world frame coordinates of best features. filtered_Features is the original ul, vl, ur, vr data for best features stored at same indices.
	print('Best Features completed')

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# Loop to calculate the average of world frame coordinates of each best-feature, basis the number of times it appears
	for feature in range(best_Features_WF.shape[1]):
	    count                               = sum(np.all(best_Features_WF[:,feature,:], axis=0)) # Count of nonzero columns - Nonzero columns give the co-ordinates of a particular feature across all timestamps
	    indices                             = np.transpose(np.nonzero(best_Features_WF[:,feature,:]))[0:count,1] # Indices of all nonzero columns in the feature matrix
	    average                             = (np.sum(best_Features_WF[:,feature,:], axis = 1))/count # Calculate the average value of World Frame co-ordinates of a particular feature appearing across all timestamps
	    mu_Features[:,feature,indices]      = average.reshape(4,1) # Store the average value in the mu matrix at the same indices at which the features are nonzero
	    mu_Features_Flatten[feature*3:(feature+1)*3, 0] = average[0:3]

	print('Mu Features completed')

	# Main time loop
	for time_step in range(features.shape[2]):

	    # One time calculations outside nt-loop
	    nt                  = sum(np.all(best_Features_WF[:,:,time_step], axis=0)) # Calculate number of features visible at a particular time-stamp
	    indices             = np.transpose(np.nonzero(best_Features_WF[:,:,time_step]))[0:nt,1] # Store the indices
	    imu_T_w             = np.linalg.inv(trajectory[:,:,time_step])
	    cam_T_w             = np.matmul(cam_T_imu, imu_T_w)
	    _cam_T_w            = np.matmul(cam_T_w, P_Transpose)

	    # Recurring Variable Definitions
	    z_Tilda             = np.zeros((4*nt, 1))
	    z_Tilda_VS          = np.zeros((4*nt, 1))
	    z_t_plus_1          = np.zeros((4*nt, 1))
	    H_t_plus_1          = np.zeros((4*nt, 3*best_Features_Count))
	    H_t_plus_1_VS       = np.zeros((4*nt, 6))
	    I_star_V            = 15*np.eye(4*nt)

	    for i in range(nt):
	        if feature_frequency[ indices[i] ] == 0:
	            pi_arg                              = np.matmul(cam_T_w, mu_Features[:,indices[i],time_step])
	            pi                                  = pi_arg/pi_arg[2]
	            z_Tilda[i*4:(i+1)*4,0]              = np.matmul( Ks, pi )
	            z_t_plus_1[i*4:(i+1)*4,0]           = filtered_Features[:,indices[i],time_step]

	            dpi_dq                              = dpi_by_dq(pi_arg)
	            temp                                = np.matmul(dpi_dq, _cam_T_w)
	            temp_H                              = np.matmul(Ks, temp)
	            H_t_plus_1[i*4:(i+1)*4, indices[i]:indices[i]+3] = temp_H
	            feature_frequency[ indices[i] ]     = 1 # Update to not consider this particular feature again

		# EKF Update Steps
	    s_t_plus_1          = ( H_t_plus_1 @ sigma_t @ H_t_plus_1.T ) + I_star_V
	    kalman_Gain         = sigma_t @ H_t_plus_1.T @ np.linalg.inv(s_t_plus_1)

	    mu_Features_Flatten = mu_Features_Flatten + np.matmul( kalman_Gain, (z_t_plus_1 - z_Tilda ) )

	    sigma_t             = ( np.eye(3*best_Features_Count) - np.matmul( kalman_Gain, H_t_plus_1) ) @ sigma_t

		# Visual Slam Steps
	    for i in range(nt):

	        m_bar                           = np.concatenate( ( mu_Features_Flatten[ indices[i]*3 : (indices[i]+1)*3, 0 ] , np.array(([1])) ), axis = 0 )
	        pi_arg                          = np.matmul(cam_T_w, m_bar)
	        pi                              = pi_arg/pi_arg[2]
	        z_Tilda_VS[i*4:(i+1)*4,0]       = np.matmul( Ks, pi ) # 4ntx1

	        dpi_dq                          = dpi_by_dq(pi_arg)
	        temp                            = np.matmul( cam_T_imu, special_dot_operator( np.matmul( imu_T_w, m_bar ) ) )
	        temp2                           = np.matmul(dpi_dq, temp)
	        H_t_plus_1_VS[i*4:(i+1)*4,:]    = -np.matmul(Ks, temp2) # 4ntx6

	    # EKF Update Steps for Visual Slam
	    s_t_plus_1                      = ( H_t_plus_1_VS @ sigma_Prediction[:,:,time_step] @ H_t_plus_1_VS.T ) + I_star_V
	    kalman_Gain                     = sigma_Prediction[:,:,time_step] @ H_t_plus_1_VS.T @ np.linalg.inv(s_t_plus_1) # 6x4nt

	    expm_argument                   = hatmap_6( np.matmul( kalman_Gain, (z_t_plus_1 - z_Tilda_VS ) ) )
	    trajectory[:,:,time_step]       = np.matmul( trajectory[:,:,time_step], expm(expm_argument) ) # 4x4

	    temp3                           = ( np.eye(6) - np.matmul( kalman_Gain, H_t_plus_1_VS ) )
	    sigma_Prediction[:,:,time_step] = temp3 @ sigma_Prediction[:,:,time_step] @ temp3.T + ( np.matmul( kalman_Gain, I_star_V ) @ kalman_Gain.T ) # 6x6

	mu_Features_Flatten = mu_Features_Flatten.reshape(best_Features_Count, 3)

	visualize_trajectory_2d(trajectory, mu_Features_Flatten[:,0], mu_Features_Flatten[:,1])
