import open3d as o3d 
import numpy as np
import tkinter as tk 
from tkinter import filedialog
from vgg_utils import * 

def select_file(*args):
    """
    Open a file dialog to select a file with a specific extension.

    Args:
        *extensions: One or more file extensions as strings (e.g., ".mp4", ".mov").
        title (str): Title for the file selection dialog.

    Returns:
        str: Absolute file path selected by the user.
    """
    root = tk.Tk()
    root.withdraw()

    # Default -> .mp4 para no joder el codigo que ya tengo. 
    if not args:
        args = (".mp4",)

    # Ensure all extensions start with a dot
    args = tuple(f".{ext.lstrip('.')}" for ext in args)

    # Create filetypes dynamically
    filetypes = [(f"{ext.upper()} files", ext) for ext in args]

    file_path = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
    
    return file_path



def estimate_normal_from_point_cloud_pca(points):
    """
    Estimate the normal vector of a point cloud using PCA.
    
    Parameters
    ----------
    points : (N, 3) ndarray
        Array of 3D points.
    
    Returns
    -------
    normal : (3,) ndarray
        Estimated normal vector of the best-fit plane.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal, centroid

def rodrigues_rotation_matrix(axis, theta):
    """
    Compute the rotation matrix using Rodrigues' rotation formula.
    
    Parameters:
        axis : (3,) array-like
            Axis of rotation (need not be normalized).
        theta : float
            Rotation angle in radians.
            
    Returns:
        R : (3, 3) ndarray
            The rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([[    0, -kz,   ky],
                  [  kz,    0, -kx],
                  [ -ky,   kx,    0]])
    I = np.eye(3)
    R = I + (K * np.sin(theta)) + (K @ K) * (1 - np.cos(theta))
    return R

def align_point_cloud_to_xy_plane(point_cloud, normal):
    """
    Rotate the point cloud so that 'normal' aligns with the Z-axis.
    
    Parameters:
        point_cloud : (N, 3) ndarray
            Array of 3D points.
        normal : (3,) array-like
            Normal vector of the best-fit plane.
    
    Returns:
        rotated_points : (N, 3) ndarray
            The rotated point cloud.
        R : (3, 3) ndarray
            The rotation matrix used to align the point cloud.
    """
    target_normal = np.array([0, 0, 1], dtype=float)
    n1 = normal / np.linalg.norm(normal)
    n2 = target_normal / np.linalg.norm(target_normal)
    dot_val = np.dot(n1, n2)
    dot_val = np.clip(dot_val, -1.0, 1.0)
    theta = np.arccos(dot_val)
    
    # Check for small rotation or 180° rotation
    if np.isclose(theta, 0.0, atol=1e-7):
        R = np.eye(3)
    elif np.isclose(theta, np.pi, atol=1e-7):
        # 180 degree rotation: choose an arbitrary perpendicular axis
        if np.allclose(n1, [0,0,1], atol=1e-7):
            axis = np.array([0,1,0])
        else:
            axis = np.cross(n1, [0,0,1])
        R = rodrigues_rotation_matrix(axis, np.pi)
    else:
        axis = np.cross(n1, n2)
        R = rodrigues_rotation_matrix(axis, theta)

    rotated_points = (R @ point_cloud.T).T
    return rotated_points, R

def create_normal_lineset(centroid, normal, length=0.5, color=[1, 0, 0]):
    """
    Create an Open3D LineSet to visualize a normal vector.
    
    Parameters:
        centroid : (3,) array-like
            The starting point of the normal (typically the point cloud centroid).
        normal : (3,) array-like
            The normal vector.
        length : float
            The length of the arrow.
        color : list of three floats
            RGB color of the line.
    
    Returns:
        lineset : open3d.geometry.LineSet
            The line set representing the normal vector.
    """
    # Ensure normal is a unit vector and scale it
    normal = normal / np.linalg.norm(normal) * length
    points = [centroid, centroid + normal]
    lines = [[0, 1]]
    colors = [color]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


if __name__ == "__main__":

    ply_file_path = select_file(".ply") 
    ply_file_folder = os.path.dirname(ply_file_path)    
    
    point_cloud =  o3d.io.read_point_cloud(ply_file_path)
    show_point_cloud(point_cloud)
    points = np.asarray(point_cloud.points)
    # estimate the normal vector of the point cloud. 

    original_normal, centroid = estimate_normal_from_point_cloud_pca(points)
    print(f"Original normal vector: {original_normal}")

    # Align the point cloud
    rotated_points, R_matrix = align_point_cloud_to_xy_plane(points, original_normal)
    rotated_normal = (R_matrix @ original_normal.reshape(-1, 1)).flatten()  # rotated normal
    print(f"Rotated normal vector: {rotated_normal}")   

    # Create Open3D point clouds
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    pcd_original.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for original

    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)
    pcd_rotated.colors = o3d.utility.Vector3dVector(point_cloud.colors)  # Use original colors  
    # Create line sets for the normals
    # Use the centroid of original point cloud as the origin for the normal arrow.
    normal_line_original = create_normal_lineset(centroid, original_normal, length=0.5, color=[1, 0, 0])
    # For the rotated point cloud, compute the new centroid after rotation.
    rotated_centroid = np.mean(rotated_points, axis=0)
    normal_line_rotated = create_normal_lineset(rotated_centroid, rotated_normal, length=0.5, color=[0, 1, 0])

    # Option 1: Visualize in separate windows (uncomment one of these blocks if you want separate views)

    # o3d.visualization.draw_geometries([pcd_original, normal_line_original],
    #                                   window_name="Original Point Cloud and Normal")
    # o3d.visualization.draw_geometries([pcd_rotated, normal_line_rotated],
    #                                   window_name="Rotated Point Cloud and Normal")

    # Option 2: Visualize both in a single window (side by side)
    # To do this, we apply a translation to one of the point clouds so they do not overlap.

    translation = np.array([3, 0, 0])  # shift rotated cloud along X-axis
    pcd_rotated.translate(translation)
    # Also shift its normal arrow accordingly
    normal_line_rotated.translate(translation)

    # Display both clouds in one view
    o3d.visualization.draw_geometries([
        pcd_original, normal_line_original, 
        pcd_rotated, normal_line_rotated
    ])