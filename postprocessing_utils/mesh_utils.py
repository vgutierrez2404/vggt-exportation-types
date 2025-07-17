import open3d as o3d 
import time
import numpy as np 
import point_cloud_utils as pcu


## first approach 

def create_mesh_from_point_cloud(point_cloud: o3d.geometry.PointCloud)-> o3d.geometry.TriangleMesh: 

    # all meshes must have the normals of their points to be computed. 
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # create the mesh 
    start_time = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10)
    mesh.compute_vertex_normals() 

    # clean parts of the mesh 
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    end_time = time.time()
    print(f"Elapsed time to compute the mesh: {end_time - start_time}")

    # Some information about the created mesh 
    print(f"The created mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles \n")    
    
    # de momento las densidades no las utilizo para nada 
    return mesh 

def fill_mesh_holes(mesh:o3d.geometry.TriangleMesh, hole_size:int=10) -> o3d.geometry.TriangleMesh: 

    # To use the fill_holes() function we need to convert the mesh to tensor mesh 

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled_mesh = tensor_mesh.fill_holes(hole_size=hole_size)

    mesh = filled_mesh.to_legacy()

    return mesh   

def simplyfy_mesh(mesh: o3d.geometry.TriangleMesh, reduction_value:float = 32) -> o3d.geometry.TriangleMesh: 
    """
    There are several methods for simplifying the vertices and triangles of a mesh. 
    The other options are available here [Mesh simplification]: https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html

    """

    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / reduction_value

    simplyfied_mesh = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,  
        contraction=o3d.geometry.SimplificationContraction.Average)

    print(f"The simplified mesh has {len(simplyfied_mesh.vertices)} vertices and {len(simplyfied_mesh.triangles)} triangles\n")    

    return simplyfied_mesh

def smooth_mesh(mesh: o3d.geometry.TriangleMesh, number_of_iterations:float = 10): 
    """
    There are several methods for smoothing the mesh. 
    The other options are available here [Mesh filtering]: https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html
    The one used is the 'Average filter' 

    """
    smoothed_mesh = mesh.filter_smooth_simple(number_of_iterations=number_of_iterations)
    smoothed_mesh.compute_vertex_normals()
    
    return smoothed_mesh


def watertight_mesh(final_watertight_mesh: o3d.geometry.TriangleMesh): 
    """
    Returns a new watertight triangle mesh based on the input mesh using the 
    point-cloud-utils library. 
    
    Inputs: 
        - mesh (o3d.geometry.TriangleMesh): original mesh 

    Output: 
        - watertight triangle mesh (o3d.geometry.TriangleMesh)
    """
    resolution = 1000

    v = np.asarray(final_watertight_mesh.vertices)
    f = np.asarray(final_watertight_mesh.triangles)

    start_time = time.time()

    vw, fw = pcu.make_mesh_watertight(v,f,resolution)

    end_time = time.time()
    print(f"Elapsed time to watertight the mesh: {end_time - start_time}")

    vertices = o3d.utility.Vector3dVector(vw)
    triangles = o3d.utility.Vector3iVector(fw)
    final_watertight_mesh = o3d.geometry.TriangleMesh(vertices,  triangles)

    final_watertight_mesh.compute_triangle_normals()
    final_watertight_mesh.compute_vertex_normals() 

    # clean parts of the mesh 
    final_watertight_mesh.remove_degenerate_triangles()
    final_watertight_mesh.remove_duplicated_triangles()
    final_watertight_mesh.remove_duplicated_vertices()
    final_watertight_mesh.remove_non_manifold_edges()

    return   final_watertight_mesh

def mesh_volume_tetrahedron(mesh):
    """
    Calculate the volume of a mesh based on the tetrahedron methos 
    from the paper [EFFICIENT FEATURE EXTRACTION FOR 2D/3D OBJECTS
    IN MESH REPRESENTATION]

    """
    # Get mesh points and faces
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)    
    # Calculate signed volumes of tetrahedra
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    
    # 
    volumes =((np.sum(np.cross(b - a, c - a) * a, axis=1)))/ 6.0
    
    return np.sum(volumes)
    

def mesh_volume_estimation(mesh:o3d.geometry.TriangleMesh): 
    """
    Based on the matlab code for estiamting the volume of a mesh. 
    https://www.mathworks.com/help/lidar/ug/estimate-stockpile-volume-of-aerial-point-cloud.html?utm_source=chatgpt.com
    """

    volume = 0.0

    for face in mesh.triangles:
        x = np.array(mesh.vertices[face[0]])
        y = np.array(mesh.vertices[face[1]])
        z = np.array(mesh.vertices[face[2]])
        partial_vol = (x[2] + y[2] + z[2]) * (x[0] * y[1] - y[0] * x[1] + y[0] * z[1] - z[0] * y[1] + z[0] * x[1] - x[0] * z[1]) / 6.0
        volume += partial_vol
    
    return volume
