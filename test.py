import numpy as np
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping


mesh1 = TriMesh('data/wolf0.off')
mesh2 = TriMesh('data/cat10.off')

# One can also specify a mesh using vertices coordinates and indexes for faces

print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\\n'
      f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')


# Initialize a FunctionalMapping object in order to compute maps
process_params = {
    'n_ev': (4,4),  # Number of eigenvalues on source and Target
    # 'landmarks': np.loadtxt('data/landmarks.txt',dtype=int)[:5],  # loading 5 landmarks
    'subsample_step': 5,  # In order not to use too many descriptors
    'descr_type': 'WKS',  # WKS or HKS
}

model = FunctionalMapping(mesh1,mesh2)
model.preprocess(**process_params,verbose=True)


# Define parameters for optimization and fit the Functional Map
fit_params = {
    'descr_mu': 1e0,
    'lap_mu': 1e-3,
    'descr_comm_mu':   1e-1,
    'orient_mu': 0
}


model.fit(**fit_params, verbose=True)


# One can access the functional map FM and vertex to vertex mapping p2p
FM = model.FM
p2p = model.p2p



# Refining is possible
# model.icp_refine()
# FM = model.FM
# p2p = model.p2p
#
# Previous information is not lost, one just need to tell which kind of functional map should be time
model.change_FM_type('classic') # Chose between 'classic', 'icp' or 'zoomout'
FM = model.FM # This is now the original FM

model.zoomout_refine(nit=50) # This refines the current model.FM, be careful which FM type is used



vis_ev=True
if vis_ev:
    canonical_color = mesh1.vertlist
    cmax = canonical_color.max(axis=0, keepdims=True)
    cmin = canonical_color.min(axis=0, keepdims=True)
    canonical_color = (canonical_color - cmin) / (cmax - cmin)


    latent_color_1 = model.project( canonical_color, mesh_ind=1)
    latent_color_2 =  model.transport( latent_color_1 )

    canonical_color2= model.decode(latent_color_2,mesh_ind=2)


    import open3d as o3d
    meshA = o3d.geometry.TriangleMesh()
    meshA.vertices = o3d.utility.Vector3dVector(mesh1.vertlist)  # [N,3]
    meshA.triangles = o3d.utility.Vector3iVector(mesh1.facelist)  # [M,3]
    meshA.vertex_colors = o3d.utility.Vector3dVector(canonical_color)
    meshA.compute_vertex_normals()



    meshB = o3d.geometry.TriangleMesh()
    meshB.vertices = o3d.utility.Vector3dVector(mesh2.vertlist + np.array([[0,-0.5,0]]))  # [N,3]
    meshB.triangles = o3d.utility.Vector3iVector(mesh2.facelist)  # [M,3]
    meshB.vertex_colors = o3d.utility.Vector3dVector(canonical_color2)
    meshB.compute_vertex_normals()

    o3d.visualization.draw_geometries ([ meshB,meshA])
