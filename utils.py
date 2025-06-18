import torch 
import numpy as np 

def uniformize(curves: torch.tensor, n: int = 200) -> torch.tensor:
    with torch.no_grad():
        l = torch.cumsum(torch.nn.functional.pad(torch.norm(curves[:,1:,:] - curves[:,:-1,:],dim=-1),[1,0,0,0]),-1)
        l = l/l[:,-1].unsqueeze(-1)
        
        sampling = torch.linspace(0,1,n).to(l.device).unsqueeze(0).tile([l.shape[0],1])
        end_is = torch.searchsorted(l,sampling)[:,1:]
        end_ids = end_is.unsqueeze(-1).tile([1,1,2])
        
        l_end = torch.gather(l,1,end_is)
        l_start = torch.gather(l,1,end_is-1)
        ws = (l_end - sampling[:,1:])/(l_end-l_start)
    
    end_gather = torch.gather(curves,1,end_ids)
    start_gather = torch.gather(curves,1,end_ids-1)
    
    uniform_curves = torch.cat([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws.unsqueeze(-1))],1)

    return uniform_curves

def preprocess_curves(curves: torch.tensor, n: int = 200) -> torch.tensor:
    
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1).unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    curves = curves/s

    # find the furthest point on the curve
    # max_idx = torch.square(curves).sum(-1).argmax(dim=1)

    # rotate curves so that the furthest point is horizontal
    # theta = -torch.atan2(curves[torch.arange(curves.shape[0]),max_idx,1],curves[torch.arange(curves.shape[0]),max_idx,0])
    theta = torch.rand([curves.shape[0]]).to(curves.device) * 2 * np.pi

    # normalize the rotation
    R = torch.eye(2).unsqueeze(0).to(curves.device)
    R = R.repeat([curves.shape[0],1,1])

    R[:,0,0] = torch.cos(theta)
    R[:,0,1] = -torch.sin(theta)
    R[:,1,0] = torch.sin(theta)
    R[:,1,1] = torch.cos(theta)

    curves = torch.bmm(R,curves.transpose(1,2)).transpose(1,2)

    return curves