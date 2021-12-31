import torch
import  numpy as np


def compute_inlier_ratio(flow_pred, flow_gt, inlier_thr=0.04, s2t_flow=None):
    inlier = torch.sum((flow_pred - flow_gt) ** 2, dim=2) < inlier_thr ** 2
    IR = inlier.sum().float() /( inlier.shape[0] * inlier.shape[1])
    return IR



def compute_nrfmr(  flow_pred,  flow_gt , recall_thr=0.04):

    nrfmr = 0.

    for i in range ( len(flow_pred)):

        dist = torch.sqrt( torch.sum( (flow_pred[i] - flow_gt[i])**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(flow_pred)

    return  nrfmr