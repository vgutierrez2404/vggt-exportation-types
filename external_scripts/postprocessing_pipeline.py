import os 
import open3d as o3d
import sys

from .postprocessing_utils.point_cloud_utils import * 
from postprocessing_utils.file_management import *
from postprocessing_utils.visualization import *
from postprocessing_utils.mesh_utils import * 
from postprocessing_utils.config_loader import ConfigLoader 
SHOW_INTERMEDIATE_RESUTS = True 
SHOW_FINAL_RESULT = True
SAVE_MESH = False

#############################################################
# Generate mesh from point cloud reconstruction with opensfm. 
#############################################################


# Read reconstruction.ply file obtained from OpenSfM 
# load the configuration
config = ConfigLoader()

ply_file_path = None 
ply_file_folder = None  
if config.load_type == "normal":    
    # aqui se debería hacer como antes. 
    ply_file_path = select_file(".ply") 
    ply_file_folder = os.path.dirname(ply_file_path)    
    
elif config.load_type=="bash": 
    if len(sys.argv) < 2: 
        print("ERROR: use python postprocessing_pipeline.py <reconstruction.ply>")
        sys.exit(1)
    ply_file_path = sys.argv[1]
    ply_file_folder = os.path.dirname(ply_file_path)

point_cloud = o3d.io.read_point_cloud(ply_file_path)
show_point_cloud(point_cloud)     
print(f"Point cloud has {len(point_cloud.points)} points")

initial_points = len(point_cloud.points) # Track points removed each filter

#region -------Points filtering------- 

# 1 stage of cleaning - Remove outliers points of the point cloud (PC from now on)  
_, point_cloud = highlight_and_clean_point_cloud(point_cloud)
print(f"------ \n Total points after 1st stage filtering: {initial_points - len(point_cloud.points)} \n------")
total_points = initial_points - len(point_cloud.points)

# 2 stage of cleaning - Remove sky points of the PC 
clean_point_cloud = remove_sky_points(point_cloud)
print(f"------ \n Total points after 2nd stage filtering: {total_points - len(clean_point_cloud.points)} \n------")

if SHOW_INTERMEDIATE_RESUTS: 
    show_point_cloud(clean_point_cloud)       

total_points = total_points - len(clean_point_cloud.points) # counter

bounding_box = get_point_cloud_bounding_box(clean_point_cloud)
# # Now we can do a filtering by boxes using the min and max values of the axis box 
filtered_point_cloud = filtering_spatial_grid_parallelized(2, bounding_box, clean_point_cloud) 
print(f"------ \n Total points after 3rd stage filtering: {total_points - len(filtered_point_cloud.points)} \n------")

if SHOW_INTERMEDIATE_RESUTS: 
    show_point_cloud(filtered_point_cloud)  

# Also, we can do an additional filtering by searching the k nearest neighbors
# for each of the points in the point cloud. This will make smoother the 
# point cloud.  
total_points = total_points - len(filtered_point_cloud.points)

kd_filtered_point_cloud = kdtree_filtering(10 , 0.5, filtered_point_cloud)
print(f"Filtered point cloud has {len(kd_filtered_point_cloud.points)} points")
print(f"------ \n Total points after 4th stage filtering: {total_points - len(kd_filtered_point_cloud.points)} \n------")

if SHOW_INTERMEDIATE_RESUTS: 
    show_point_cloud(kd_filtered_point_cloud)

#endregion 


#region -------Floor fitting-------

floored_point_cloud, only_floor_point_cloud = fit_plane_svd(kd_filtered_point_cloud, 5000)
if SHOW_INTERMEDIATE_RESUTS: 
    show_point_cloud(floored_point_cloud)

#endregion 

#region -------Mesh creation-------
pseudo_closed_point_cloud, pseudo_closed_mesh = fit_plane_top_bottom(kd_filtered_point_cloud, add_roof=True, add_floor=True, shrink_factor=0.75)
o3d.io.write_point_cloud(os.path.join(ply_file_folder, "filtered_point_cloud.ply"), pseudo_closed_point_cloud)

mesh_smp  = simplyfy_mesh(pseudo_closed_mesh, reduction_value=64)
if SHOW_INTERMEDIATE_RESUTS: 
    show_mesh(mesh_smp)
    
mesh_smooth= smooth_mesh(mesh_smp, number_of_iterations=3)
if SHOW_INTERMEDIATE_RESUTS: 
    show_mesh(mesh_smooth)

# This approach for filling the mesh won't be used since the open3d function 
# does not work properly. 
"""
filled_stockpile_mesh = fill_mesh_holes(pseudo_closed_mesh, hole_size=50)
if SHOW_INTERMEDIATE_RESUTS:
    show_mesh(filled_stockpile_mesh)
"""

final_watertight_mesh = watertight_mesh(mesh_smooth)
show_mesh(final_watertight_mesh)


if SAVE_MESH: 
    mesh_file_path = os.path.join(ply_file_folder, "final_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file_path, final_watertight_mesh)

#endregion


#region --------Calculate volume---------

if final_watertight_mesh.is_watertight(): 
    print(f"Volume of the mesh {o3d.geometry.TriangleMesh.get_volume(final_watertight_mesh)}")
else: 
    print("The mesh is not watertight")

print(f"Tetrahedron method for the volume of the mesh {mesh_volume_tetrahedron(final_watertight_mesh)}")

print(f"matlab volume of a mesh: {mesh_volume_estimation(final_watertight_mesh)}")  
#endregion

