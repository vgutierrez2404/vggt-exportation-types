import open3d as o3d
import numpy as np 


# python pasa la referencia del objeto que se pasa como argumento asi que no
# pasa nada por que pase la point cloud. 
def show_point_cloud(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Visualizes a 3D point cloud using Open3D with axis-aligned bounding boxes.

    This function takes an Open3D PointCloud object, calculates its axis-aligned bounding box,
    and displays the point cloud along with the bounding boxes in an Open3D visualizer.
    The background is set to black for better visibility.

    Input:
        point_cloud: The point cloud to be visualized.

    """   

    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("Expected an Open3D PointCloud object.")

    # Get axis limits
    min_bound = np.min(np.asarray(point_cloud.points), axis=0)
    max_bound = np.max(np.asarray(point_cloud.points), axis=0)

    # Create
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



def show_mesh(meshes:list[o3d.geometry.TriangleMesh], convert_tensor_to_legacy:bool=False) -> None:
    """
    Checks the list of meshes and plots them 

    Input: 
        - meshes (list[o3d.t.geometry.TriangleMesh or o3d.geometry.TriangleMesh]): input meshes 
        - convert_tensor_to_legacy (bool): bool to convert tensor based meshes into o3d.geometry.TriangleMesh meshes
    """
    single_mesh = True
    tensor_based = []
    normal_mesh = []
    if isinstance(meshes,list): 
        single_mesh = False
        for i, mesh in enumerate(meshes): 
            if isinstance(mesh, o3d.geometry.TriangleMesh): 
                print(f"Mesh {i} is modern open3d TriangleMesh")
                normal_mesh.append(mesh)
        
            elif isinstance(mesh, o3d.t.geometry.TriangleMesh): 
                print(f"Mesh {i} is tensor based TriangleMesh")
                tensor_based.append(mesh)
        
                if convert_tensor_to_legacy: 
                    mesh = mesh.to_legacy()
                    tensor_based.pop() # remove the last element of the list
                    normal_mesh.append(mesh)
    else: 
        normal_mesh.append(meshes) 

    if normal_mesh:  # check if the list is not empty
        o3d.visualization.draw_geometries(normal_mesh, mesh_show_back_face=True, point_show_normal=True)
    
    if tensor_based or (single_mesh and isinstance(meshes, o3d.t.geometry.TriangleMesh)): # check if the list is not empty
        o3d.visualization.draw([{'name':'Tensor meshes', 'geometry': tensor_based}])

