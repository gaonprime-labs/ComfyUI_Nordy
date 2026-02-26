import torch 
import torch.nn.functional as F

def laplacian_variance(gray_bchw):
    device = gray_bchw.device
    dtype = gray_bchw.dtype

    lap_kernel = torch.tensor(
        [[[[0, -1, 0],
           [-1,  4, -1],
           [0, -1, 0]]]],
        device=device,
        dtype=dtype
    )

    resp = F.conv2d(gray_bchw, lap_kernel, padding=1)  
    resp = resp.view(resp.size(0), -1)                 
    return resp.var(dim=1)    