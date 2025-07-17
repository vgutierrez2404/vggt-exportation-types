from scipy.interpolate import interp1d
import numpy as np
import csv



def slerp(q1, q2, t):
    # Function for performing spherical linear interpolation between two quaternions

    # Normalize the quaternions
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    q2 /= np.linalg.norm(q2, axis=1, keepdims=True)

    dot_product = np.sum(q1 * q2, axis=1)

    # If the quaternions are opposite, perform linear interpolation
    opposite_indices = dot_product < 0.0
    q1[opposite_indices] = -q1[opposite_indices]
    dot_product[opposite_indices] = -dot_product[opposite_indices]

    # Define a threshold to avoid division by zero
    threshold_indices = dot_product > 0.9995
    result = np.where(
        threshold_indices[:, None],
        q1 + t * (q2 - q1),
        (np.cos(t * np.arccos(dot_product[:, None])) - dot_product[:, None] * np.sin(
            t * np.arccos(dot_product[:, None])) / np.sin(np.arccos(dot_product[:, None]))) * q1 +
        (np.sin(t * np.arccos(dot_product[:, None])) / np.sin(np.arccos(dot_product[:, None]))) * q2
    )

    return result / np.linalg.norm(result, axis=1, keepdims=True)



def interpolate_coordinates(csv_file:str, orientation_file:str, timestamp_frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    def interpolate_values(timestamps_in, values, target_time):

        interpolator = interp1d(timestamps_in, values)

        return interpolator(target_time)

    def find_nearest_values(time, time2):

        lower_values = time[np.searchsorted(time, time2) - 1]
        upper_values = time[np.searchsorted(time, time2)]

        upper_values[time2 >= time[-1]] = None
        lower_values[time2 <= time[0]] = None

        return lower_values, upper_values

    def interpolate_values(timestamps_in, values, target_time):

        interpolator = interp1d(timestamps_in, values, fill_value="extrapolate") # interp1d is legacy 
        interpolated_values = interpolator(target_time)
        interpolated_values = np.where(
            np.logical_or(target_time < np.min(timestamps_in), target_time > np.max(timestamps_in)), np.nan,
            interpolated_values)
        
        return interpolated_values

    # Read Location file
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data_in_csv = list(reader)

    timestamps = [float(row['time']) for row in data_in_csv]
    longitudes = [float(row['longitude']) for row in data_in_csv]
    latitudes = [float(row['latitude']) for row in data_in_csv]
    altitude =  [float(row['altitude']) for row in data_in_csv]

    # Interpolar longitud, latitud y altitud para cada valor en timestamp_frame
    interpolated_longitudes = interpolate_values(timestamps, longitudes, timestamp_frame)
    interpolated_latitudes = interpolate_values(timestamps, latitudes, timestamp_frame)
    interpolated_altitudes = interpolate_values(timestamps, altitude, timestamp_frame)

    if orientation_file:
        # Read orientation file
        with open(orientation_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data_in_csv = list(reader)

        timestamps = [float(row['time']) for row in data_in_csv]
        qz = [float(row['qz']) for row in data_in_csv]
        qy = [float(row['qy']) for row in data_in_csv]
        qx = [float(row['qx']) for row in data_in_csv]
        qw = [float(row['qw']) for row in data_in_csv]

        # Create a matrix with the interpolated quaternions
        interpolated_quaternions = np.empty((len(timestamp_frame), 4))
        for i in range(1, len(timestamp_frame)):
            index = np.searchsorted(timestamps, timestamp_frame[i], side='right') - 1
            t = ( timestamps[index]- timestamp_frame[i]) / (timestamps[index] - timestamps[index+1])      
            #t = np.clip(t, 0, 1)  # Ensure t is within the range [0, 1]

            q1 = np.column_stack((qz[index], qy[index], qx[index], qw[index]))
            q2 = np.column_stack((qz[index+1], qy[index+1], qx[index+1], qw[index+1]))

            interpolated_quaternions[i] = slerp(q1, q2, t)

    else:
        interpolated_quaternions = None

    return interpolated_longitudes, interpolated_latitudes, interpolated_altitudes, interpolated_quaternions
