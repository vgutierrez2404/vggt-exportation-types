import random 
import os 
import logging
import re 
import math 
import open3d as o3d
import numpy as np 
from datetime import datetime
import torch 

def add_header(num_vertices:int, comment:str = "format ascii 1.0" ) -> list[str]: 
    """define the header of a ply file from: https://people.math.sc.edu/Burkardt/data/ply/ply.txt"""
    # normal header for the function. 
    ply_header = ["ply", 
                  comment, 
                  "element vertex {}".format(num_vertices),
                  "property float32 x",
                  "property float32 y",
                  "property float32 z",
                  "property uint8 red",
                  "property uint8 green",
                  "property uint8 blue", 
                  "end_header"]

    return ply_header

def add_data(points_3d) -> list[str]: 
    """simple: add points and colors data to the file. this could be more complex 
     if we want to add the camera positions ... not for the moment"""
    ply_points = []

    for point in points_3d: 
        x, y, z = point["xyz"]
        r, g, b = point["rgb"]
        row = f"{x} {y} {z} {r} {g} {b}"
        
        ply_points.append(row)
    
    return ply_points

def write_points3d_ply(filename, points_3d): 

    header = add_header(len(points_3d))
    ply_points = add_data(points_3d)
    file_srt = "\n".join(header + ply_points + [""])

    with open(filename, "w") as file: 
        file.write(file_srt)

def split_and_sample_frames(folder_path: str, number_datasets: int) -> list[str]:
    """
    from the dataset, split it in number_frames datasets and select a random frame from each of the subdatasets. 
    """

    all_frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Compute the size of each subdataset
    total_frames = len(all_frames)
    subset_size = math.ceil(total_frames / number_datasets)  # Round up to ensure all frames are included

    subdatasets = [all_frames[i * subset_size : (i + 1) * subset_size] for i in range(number_datasets)]
    selected_frames = [random.choice(subset) for subset in subdatasets if subset]

    return selected_frames


def extract_frame_numbers(frame_paths):
    """Extracts the frame number from the file path."""
    variations = ["frame", "img"] 
    for var in variations:  
        frame_numbers = []
        for frame in frame_paths:
            match = re.search(rf"{var}_(\d+)\.jpg", os.path.basename(frame))
            if match:
                frame_numbers.append(match.group(1))
        if frame_numbers:
            break
    return frame_numbers

def log_reconstructions(info:dict): 

    for key,value in info.items(): 
        logging.info(f"{key}: {value}")
    logging.info("Reconstruction completed.\n")

def show_point_cloud(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Visualizes a 3D point cloud using Open3D with axis-aligned bounding boxes.

    This function takes an Open3D PointCloud object, calculates its axis-aligned bounding box,
    and displays the point cloud along with the bounding boxes in an Open3D visualizer.
    The background is set to black for better visibility.

    Input:
        point_cloud: The point cloud to be visualized.

    """   

    print(f"point cloud has {point_cloud.points} points")

    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("Expected an Open3D PointCloud object.")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()

    return 


def get_gpu_memory_usage():
    """Returns the GPU memory usage in MB for the current PyTorch device."""
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        return mem_allocated, mem_reserved  # Returns two values
    else:
        return 0, 0  # No GPU available


# Function to parse the log.txt file
def parse_log_file(log_path):
    log_data = {
        "total_frames": None,
        "num_datasets": None,
        "selected_frames": [],
        "confidence_threshold": None,
        "elapsed_time": None,
        "memory_allocated_mb": None,
        "memory_reserved_mb": None,
        "reconstruction_status": None
    }

    with open(log_path, 'r') as f:
        for line in f:
            # Extract Total Frames
            if "Total Frames" in line:
                log_data["total_frames"] = int(re.search(r"Total Frames: (\d+)", line).group(1))
            # Extract Number of Datasets
            elif "Number of Datasets" in line:
                log_data["num_datasets"] = int(re.search(r"Number of Datasets: (\d+)", line).group(1))
            # Extract Selected Frames
            elif "Selected Frames" in line:
                frames = re.search(r"Selected Frames: ([\d,]+)", line).group(1)
                log_data["selected_frames"] = frames.split(",")
            # Extract Confidence Threshold
            elif "Confidence threshold" in line:
                log_data["confidence_threshold"] = float(re.search(r"Confidence threshold: ([\d.]+)", line).group(1))
            # Extract Elapsed Time
            elif "Elapsed time for reconstruction" in line:
                log_data["elapsed_time"] = float(re.search(r"Elapsed time for reconstruction: ([\d.]+)", line).group(1))
            # Extract Memory Change (Allocated MB)
            elif "Memory Change (Allocated MB)" in line:
                log_data["memory_allocated_mb"] = float(re.search(r"Memory Change \(Allocated MB\): ([\d.]+)", line).group(1))
            # Extract Memory Change (Reserved MB)
            elif "Memory Change (Reserved MB)" in line:
                log_data["memory_reserved_mb"] = float(re.search(r"Memory Change \(Reserved MB\): ([\d.]+)", line).group(1))
            # Extract Reconstruction Status
            elif "Reconstruction completed" in line:
                log_data["reconstruction_status"] = "Completed"

    return log_data

def print_log_data(log_data, subfolder):
    print(f"\nLog Data for Subfolder: {subfolder}")
    print(f"Total Frames: {log_data['total_frames']}")
    print(f"Number of Datasets: {log_data['num_datasets']}")
    print(f"Selected Frames: {', '.join(log_data['selected_frames'])}")
    print(f"Confidence Threshold: {log_data['confidence_threshold']}")
    print(f"Elapsed Time for Reconstruction: {log_data['elapsed_time']} seconds")
    print(f"Memory Change (Allocated MB): {log_data['memory_allocated_mb']} MB")
    print(f"Memory Change (Reserved MB): {log_data['memory_reserved_mb']} MB")
    print(f"Reconstruction Status: {log_data['reconstruction_status']}")

def parse_subfolder_timestamp(subfolder_name):
    try:
        # Replace underscore with space and parse the timestamp
        timestamp_str = subfolder_name.replace('_', ' ')
        # Parse the string "YYYY-MM-DD HH-MM-SS" into a datetime object
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H-%M-%S')
    except ValueError:
        # If parsing fails (e.g., subfolder name doesn't match the format), return None
        return None


def show_point_cloud(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Visualizes a 3D point cloud using Open3D with axis-aligned bounding boxes.

    This function takes an Open3D PointCloud object, calculates its axis-aligned bounding box,
    and displays the point cloud along with the bounding boxes in an Open3D visualizer.
    The background is set to black for better visibility.

    Input:
        point_cloud: The point cloud to be visualized.

    """   

    print(f"point cloud has {point_cloud.points} points")
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("Expected an Open3D PointCloud object.")

    # Get axis limits
    min_bound = np.min(np.asarray(point_cloud.points), axis=0)
    max_bound = np.max(np.asarray(point_cloud.points), axis=0)

    # Create bounding box
    x_axis_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Create the visualization object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(x_axis_box)

    # Config black background
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # Visualizar la escena
    vis.run()

    return 

