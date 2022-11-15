import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import *
from process_lidar_data import process_Lidar_Data


# Function Definitions

def initialize_Lidar():

    timestamp, data = read_data_from_csv('data/sensor_data/lidar.csv')

    return timestamp, data

def initialize_Encoder():

    timestamp, data = read_data_from_csv('data/sensor_data/encoder.csv')

    return timestamp, data

def initialize_FOG():

    timestamp, data = read_data_from_csv('data/sensor_data/fog.csv')

    return timestamp, data

def sync_Encoder_FOG_Data(_data_fog, _timestamp_encoder):

    _delta_yaw           = np.zeros(116048)
    _dt                  = np.zeros(116048)

    for i in range( len(_timestamp_encoder) - 1 ):
        _delta_yaw[i]    = sum( _data_fog[ ((i - 1)*10 + 1) : (i*10 + 1), 2 ] )
        _dt[i]           = (_timestamp_encoder[i+1] - _timestamp_encoder[i] ) * 10**(-9) # Conversion from nanoseconds to seconds

    return _delta_yaw, _dt

def add_Noise(_particle_state, _particle_count, _dt):

    temp_State_Array = np.zeros((_particle_count, 3))

    for i in range(_particle_count):

        linear_velocity_noise       = np.random.normal(0, 0.5)
        angular_velocity_noise      = np.random.normal(0, 0.05)
        temp_State_Array[i,0]       = _particle_state[i,0] + (linear_velocity_noise * _dt)
        temp_State_Array[i,1]       = _particle_state[i,1] + (linear_velocity_noise * _dt)
        temp_State_Array[i,2]       = _particle_state[i,2] + (angular_velocity_noise * _dt)

    return temp_State_Array


def initialize_Map(MAP):

    MAP['res']      = 1 # Meters
    MAP['xmin']     = -100  # Meters
    MAP['ymin']     = -1200 # Meters
    MAP['xmax']     = 1300 # Meters
    MAP['ymax']     = 100  # Meters
    MAP['sizex']    = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # Cells
    MAP['sizey']    = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map']      = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32)

    _x_im           = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) # x-positions of each pixel of the map
    _y_im           = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) # y-positions of each pixel of the map

    _x_range        = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])
    _y_range        = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])

    return MAP, _x_im, _y_im, _x_range, _y_range

def calculate_Map_Correlation(_data_lidar, _particle_count, _particle_state, _particle_weights, MAP, _x_im, _y_im, _x_range, _y_range):

    correlation         = np.zeros(_particle_count)
    #im                  = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.5).astype(np.int)

    for i in range(_particle_count):
        _xs0, _ys0      = process_Lidar_Data(_data_lidar, _particle_state[i])
        Y               = np.stack((_xs0, _ys0))
        corr            = mapCorrelation(MAP['map'], _x_im, _y_im, Y, _x_range, _y_range)
        correlation[i]  = np.max(corr)

    # Update Particle Weights using Softmax Function
    max_corr                    = np.max(correlation)
    beta                        = np.exp(correlation - max_corr)
    ph                          = beta / beta.sum()
    _particle_weights           = _particle_weights * ( ph / np.sum(_particle_weights * ph) )

    # Find the particle which matches best with the Map
    _position_matched_particle  = np.argmax(_particle_weights)
    _state_matched_particle     = _particle_state[_position_matched_particle, :]

    return _state_matched_particle

def update_Map(MAP, _xs0, _ys0, _particle_state, _log_odds_ratio):

    # convert from meters to cells
    xis             = np.ceil((_xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis             = np.ceil((_ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    x_origin        = np.ceil((_particle_state[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_origin        = np.ceil((_particle_state[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    output = np.zeros((2,0))

    for i in range(len(xis)):
        bresenham_output    = bresenham2D(x_origin, y_origin, xis[i], yis[i])
        output              = np.hstack((output, bresenham_output))

    bresenham_x     = output[0,:].astype(int)
    bresenham_y     = output[1,:].astype(int)

    indGood = np.logical_and(np.logical_and(np.logical_and((bresenham_x > 1), (bresenham_y > 1)), (bresenham_x < MAP['sizex'])), (bresenham_y < MAP['sizey']))

    # Update Map using log-odds ratio

    # Decrease the log-odds for free cells
    MAP['map'][ bresenham_x[indGood] , bresenham_y[indGood] ] += _log_odds_ratio

    # Increase the log-odds for occupied cells
    for i in range(len(xis)):
        if (xis[i] > 1) and (xis[i] < MAP['sizex']) and yis[i] > 1 and (yis[i] < MAP['sizey']):
            MAP['map'][ xis[i] , yis[i] ] -= _log_odds_ratio

    # Clip Map to the maximum and minimum values given
    MAP['map'] = np.clip(MAP['map'], -10*_log_odds_ratio, 10*_log_odds_ratio)

    return MAP

def superimpose_Trajectory(_trajectory, MAP):

    _x        = np.ceil((_trajectory[:,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    _y        = np.ceil((_trajectory[:,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    return _x, _y

def plot_Map(MAP, _x, _y):

    # Show Occupancy Grid Map
    plt.imshow(MAP['map'], cmap='gray')

    # Extract PMF for Binary Map
    binary_Map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(np.int)

    # Show the Binary Map
    #plt.imshow(binary_Map, cmap = 'gray')

    # Plot Trajectory on the Map
    plt.plot(_y, _x, color='orangered', linewidth=0.6)

    # Map Details
    plt.title("Occupancy Grid Map")
    plt.xlabel("x grid-cell coordinates")
    plt.ylabel("y grid-cell coordinates")

    plt.show(block=True)

    return 0
