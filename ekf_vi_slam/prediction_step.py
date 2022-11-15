import numpy as np
from pr3_utils import *
from utility import *
from scipy.linalg import expm

def prediction_Step( _t, _linear_velocity, _angular_velocity ):

    # Function Variable Definitions
    _trajectory     = np.zeros((4,4,_t.shape[1])) # Each pose will be 4x4
    mu              = np.zeros((4,4,_t.shape[1])) # Each mu will be 4x4
    sigma           = np.zeros((6,6,_t.shape[1])) # Each sigma will be 6x6
    mu[:,:,0]       = np.eye(4) # Initiliazing the mu as Identity matrix of 4x4
    sigma[:,:,0]    = np.zeros((6,6)) # Initiliazing the sigma as Identity matrix of 6x6
    W               = np.diag([0.5, 0.5, 0.5, 0.05, 0.05, 0.05])

    # Calculating delta_mu for the first pose
    del_mu              = np.diag( np.random.normal( 0, sigma[:,:,0] ) )
    _trajectory[:,:,0]  = np.matmul( mu[:,:,0], expm( hatmap_6( del_mu ) ) )

    # Prediction Loop
    for step in range(_t.shape[1]-1):
        tau_t                   = _t[0,step + 1] - _t[0,step] # scalar value
        u_t                     = np.concatenate(( _linear_velocity[:,step], _angular_velocity[:,step] ), axis = 0) # 6x1
        w_t                     = np.diag( np.random.normal( 0, W ) ) # 6x1

        mu[:,:,step+1]          = np.matmul( mu[:,:,step], expm( tau_t * hatmap_6(u_t) )) # Mu - Predicted - Dimension: 4x4xtime
        sigma[:,:,step+1]       = np.matmul( expm( -tau_t * curly_hatmap(u_t) ), sigma[:,:,step] )
        sigma[:,:,step+1]       = np.matmul( sigma[:,:,step+1], expm( -tau_t * curly_hatmap(u_t) ).T ) + W # Sigma - Predicted - Dimension: 6x6xtime

        del_mu                  = np.matmul( expm( -tau_t * curly_hatmap(u_t) ), del_mu ) + w_t # 6x1
        _trajectory[:,:,step+1] = np.matmul( mu[:,:,step+1], expm( hatmap_6( del_mu ) ) ) #Dimension: 4x4xtime

    return _trajectory, sigma
