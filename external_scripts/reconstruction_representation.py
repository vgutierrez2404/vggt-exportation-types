import open3d as o3d 
import os 
from external_scripts.vgg_utils import * 

EXTERNAL_SCRIPT_PATH = "/home/gaps-canteras-u22/Documents/repos/essentials-processing/postprocessing_pipeline.py"
output_dir = "/home/gaps-canteras-u22/Documents/repos/vggt/images/VID_20240321_100947/images/selected_frames/output_dir"
subfolders = os.listdir(output_dir)

subfolders_with_time = []
for subfolder in subfolders:
    subfolder_path = os.path.join(output_dir, subfolder)
    if os.path.isdir(subfolder_path):  # Ensure it's a directory
        timestamp = parse_subfolder_timestamp(subfolder)
        if timestamp:  # Only include subfolders with valid timestamps
            subfolders_with_time.append((subfolder, timestamp))
subfolders_with_time.sort(key=lambda x: x[1]) 
sorted_subfolders = [subfolder for subfolder, _ in subfolders_with_time] 

points3d_paths = []

for subfolder in sorted_subfolders: 
    subfolder_path = os.path.join(output_dir, subfolder)

    if os.path.isdir(subfolder_path):
        log_path = os.path.join(subfolder_path, "log.txt")
        log_data = parse_log_file(log_path)
        print_log_data(log_data, subfolder_path)
        points3d_path = os.path.join(subfolder_path, "points3D.ply")
    
        if os.path.isfile(points3d_path):
            points3d_paths.append(points3d_path)
            point_cloud = o3d.io.read_point_cloud(points3d_path)
            show_point_cloud(point_cloud)
