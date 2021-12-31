import numpy as np
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping
from _4DMatch import _4DMatch
from _4dmatch_metric import *
import torch
dataset = _4DMatch('test')

IR = 0.
NFMR = 0.


for i in range (len(dataset)):

    src_pcd, tgt_pcd, sflow, _,_, metric_index = \
        dataset.__getitem__(i, debug=False)


    mesh1 = TriMesh( vertices=tgt_pcd )
    mesh2 = TriMesh( vertices=src_pcd )


    process_params = {
        'n_ev': (4,4),  # Number of eigenvalues on source and Target
        # 'landmarks': np.loadtxt('data/landmarks.txt',dtype=int)[:5],  # loading 5 landmarks
        'subsample_step': 5,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
    }
    model = FunctionalMapping(mesh1,mesh2)
    model.preprocess(**process_params,verbose=True)


    fit_params = {
        'descr_mu': 1e0,
        'lap_mu': 1e-3,
        'descr_comm_mu':   1e-1,
        'orient_mu': 0
    }
    model.fit(**fit_params, verbose=True)


    # One can access the functional map FM and vertex to vertex mapping p2p
    model.change_FM_type('classic') # Chose between 'classic', 'icp' or 'zoomout'
    FM = model.FM # This is now the original FM
    model.zoomout_refine(nit=50) # This refines the current model.FM, be careful which FM type is used
    p2p=model.p2p

    gt_flow = torch.from_numpy( sflow[None] )
    pred_flow =torch.from_numpy( tgt_pcd[p2p] - src_pcd )
    i_rate = compute_inlier_ratio(pred_flow, gt_flow , inlier_thr=0.04)

    anchor_points = src_pcd[metric_index]
    pred_flow = tgt_pcd [ p2p[metric_index] ] - anchor_points
    pred_flow = torch.from_numpy( pred_flow[[None]])
    gt_flow = torch.from_numpy( sflow[metric_index][None] )
    nfmr = compute_nrfmr(pred_flow,gt_flow)

    IR += i_rate
    NFMR += nfmr
    print(i, "/", len(dataset), "IR:", IR / (i + 1), "NFMR:", NFMR / (i + 1))

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
        meshA = o3d.geometry.PointCloud()
        meshA.points = o3d.utility.Vector3dVector(mesh1.vertlist)  # [N,3]
        meshA.colors = o3d.utility.Vector3dVector(canonical_color)



        meshB = o3d.geometry.PointCloud()
        meshB.points = o3d.utility.Vector3dVector(mesh2.vertlist + np.array([[0,-0.5,0]]))  # [N,3]
        meshB.colors = o3d.utility.Vector3dVector(canonical_color2)

        o3d.visualization.draw_geometries ([ meshB,meshA])