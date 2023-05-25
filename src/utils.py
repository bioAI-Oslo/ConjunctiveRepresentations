import torch
import numpy as np
import matplotlib.pyplot as plt

class SimpleDatasetMaker(object):
    # Simple test dataset maker; square box + bounce off walls

    def __init__(self, chamber_size = 1, stddev = 4*np.pi, rayleigh_scale = 0.025):
        self.chamber_size = chamber_size
        self.stddev = stddev
        self.rayleigh_scale = rayleigh_scale
        
    def bounce(self, r, v):
        outside = np.abs(r + v) >= self.chamber_size
        v[outside] = -v[outside]
        return v

    def generate_dataset(self, samples, timesteps, device = "cpu"):
        r = np.zeros((samples, timesteps, 2))

        s = np.random.rayleigh(self.rayleigh_scale, (samples, timesteps))
        prev_hd = np.random.uniform(0, 2*np.pi, samples)
        r[:,0] = np.random.uniform(-self.chamber_size, self.chamber_size, (samples, 2)) 

        for i in range(timesteps - 1):
            hd = np.random.vonmises(prev_hd, self.stddev, samples)
            prop_v = s[:,i,None]*np.stack((np.cos(hd), np.sin(hd)),axis=-1)
            v = self.bounce(r[:,i], prop_v) 
            prev_hd = np.arctan2(v[:,1], v[:,0])
            r[:,i+1] = r[:,i] + v # dt = 1

        v = np.diff(r, axis = 1)
        return torch.tensor(r.astype('float32'), device = device), torch.tensor(v.astype('float32'), device = device)

def weighted_kde(mu, w, bw = 1):
    # x.shape = N, 2
    # mu.shape = M, 2
    # w.shape = N
    # 1, M, 2 - N, 1, 2 --> N, M, 2
    # ---> N, M ---> N
    def kernel(x): 
        d = np.sum((mu[None] - x[:,None])**2,axis=-1)
        return np.sum(w*np.exp(-0.5/bw**2*d), axis = 1)
    return kernel


def ratemap_collage(ratemaps, cols = 5 ,figsize = (5,5), **kwargs):
    """ plot collage of ratemaps

    Args:
        ratemaps (np ndarray): array of shape (N, bins, bins). 
            Should contain N ratemaps, each of shape (bins, bins)
        cols (int, optional): Number of units in each row. Defaults to 5.
        figsize (tuple, optional): Size of figure. Defaults to (5,5).
    """
    ratio = len(ratemaps) // cols 
    rows = ratio if ((len(ratemaps) % cols) == 0) else ratio + 1
    fig, axs = plt.subplots(cols, rows, figsize = figsize, **kwargs)

    for i in range(len(ratemaps)):
        axs[i//cols, i%cols].imshow(ratemaps[i].T, origin = "lower")
        axs[i//cols, i%cols].axis("off")
    
    return fig, axs
