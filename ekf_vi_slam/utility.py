import numpy as np
from scipy.linalg import expm

def hatmap(x_vector):

    # Create hatmap of 3x1 input vector
    # Output matrix = 3x3

    x_hatmap        = np.zeros((3,3))
    x_hatmap[0,1]   = -x_vector[2]
    x_hatmap[0,2]   = x_vector[1]
    x_hatmap[1,0]   = x_vector[2]
    x_hatmap[1,2]   = -x_vector[0]
    x_hatmap[2,0]   = -x_vector[1]
    x_hatmap[2,1]   = x_vector[0]

    return x_hatmap

def hatmap_6(x_vector):

    # Create hatmap of 6x1 input vector
    # Output matrix = 4x4

    x_hatmap_6              = np.zeros((4,4))
    x_hatmap_6[0:3, 0:3]    = hatmap(x_vector[3:6])
    x_hatmap_6[0:3, 3]      = x_vector[0:3].reshape(3,)

    return x_hatmap_6

def curly_hatmap(x_vector):

    # Creat curly hatmap of 6x1 input vector
    # Output matrix = 6x6

    x_curly_hatmap              = np.zeros((6,6))
    x_curly_hatmap[0:3, 0:3]    = hatmap(x_vector[3:6])
    x_curly_hatmap[0:3, 3:6]    = hatmap(x_vector[0:3])
    x_curly_hatmap[3:6, 3:6]    = hatmap(x_vector[3:6])

    return x_curly_hatmap

def formulateKs(_K, _baseline):

    _Ks             = np.zeros((4,4))
    _Ks[0:3, 0:3]   = _K
    _Ks[2,:]        = _Ks[0,:]
    _Ks[3,:]        = _Ks[1,:]
    _Ks[2,3]        = -_Ks[0,0] * _baseline

    return _Ks

def compute_WF_Coord(_features, _Ks, _imu_T_cam, _trajectory):

    _WF_Coord       = np.zeros(_features.shape)
    disparity       = _features[0,:] - _features[2,:]
    fsub            = -_Ks[2,3]
    fsu             = _Ks[0,0]
    fsv             = _Ks[1,1]
    cu              = _Ks[0,2]
    cv              = _Ks[1,2]
    pose            = np.matmul(_trajectory, _imu_T_cam)

    for i in range(_features.shape[1]):
        if disparity[i] != 0:
            _WF_Coord[2,i]  = fsub / disparity[i]
            _WF_Coord[0,i]  = ( ( _features[0,i] - cu ) * _WF_Coord[2,i] ) / fsu
            _WF_Coord[1,i]  = ( ( _features[1,i] - cv ) * _WF_Coord[2,i] ) / fsv
            _WF_Coord[3,i]  = 1
            _WF_Coord[:,i]  = np.matmul( pose, _WF_Coord[:,i] )

    return _WF_Coord

def best_Features(_features_wf, _features, _count):

    _best_features      = np.zeros((4 , _count, _features_wf.shape[2]))
    _filtered_features  = np.zeros((4 , _count, _features_wf.shape[2]))
    count_array         = np.zeros((_features_wf.shape[1], 2), dtype = np.int32)

    for feature in range(_features_wf.shape[1]):
        count_array[feature,0]  = feature
        count_array[feature,1]  = sum(np.all(_features_wf[:,feature,:], axis=0))

    count_array         = np.flip(count_array[count_array[:, 1].argsort()])
    count_array         = count_array[0:_count]
    indices             = (count_array[:,1])
    indices             = np.sort(indices)
    _best_features      = _features_wf[:,indices,:]
    _filtered_features  = _features[:,indices,:]

    return _best_features, _filtered_features

def dpi_by_dq(q):

    # Take derivative of 4x1 input vector
    # Output matrix = 4x4

    _q      = np.zeros((4,4)) # Variable definition for Derivative of q
    _q[0,0] = 1
    _q[1,1] = 1
    _q[3,3] = 1
    _q[0,2] = -q[0]/q[2]
    _q[1,2] = -q[1]/q[2]
    _q[3,2] = -q[3]/q[2]
    _q      = _q / q[2]

    return _q

def special_dot_operator(x_vector):

    # Special Dot character of 4x1 input vector
    # Output matrix = 4x6

    x_output            = np.zeros((4,6))
    s                   = x_vector[0:3]
    s_hat               = hatmap(s)
    x_output[0:3,0:3]   = np.eye(3)
    x_output[0:3,3:6]   = -s_hat

    return x_output
