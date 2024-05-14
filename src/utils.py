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


def ratemap_collage(ratemaps, cols=5, figsize=(5, 5), cmap="viridis", vmin=None, vmax=None, **kwargs):
    """ plot collage of ratemaps

    Args:
        ratemaps (np ndarray): array of shape (N, bins, bins). 
            Should contain N ratemaps, each of shape (bins, bins)
        cols (int, optional): Number of units in each row. Defaults to 5.
        figsize (tuple, optional): Size of figure. Defaults to (5,5).
        cmap (str, optional): Matplotlib colormap. Defaults to "viridis".

    """
    ratio = len(ratemaps) // cols 
    rows = ratio if ((len(ratemaps) % cols) == 0) else ratio + 1
    fig, axs = plt.subplots(cols, rows, figsize=figsize, **kwargs)

    for i in range(len(ratemaps)):
        axs[i // cols, i % cols].imshow(ratemaps[i].T, origin="lower", interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i // cols, i % cols].axis("off")
    
    return fig, axs

def spatial_correlation(a, b):
    """    
        Compute spatial correlation (Leutgeb et al. 2005)

        Takes in arrays of ratemaps of shape (n_cells, binx, biny)
        returns distribution of Pearson correlations
    """    
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)

    sum_a = np.sum(a_flat, axis = -1) 
    sum_b = np.sum(b_flat, axis = -1)
    mask = np.logical_and(sum_a > 0, sum_b > 0) # only include units with nonzero response
    
    a_flat = a_flat[mask]
    b_flat = b_flat[mask]

    corr = np.zeros(a_flat.shape[0])
    for i in range(a_flat.shape[0]):
        corr[i] = np.corrcoef(a_flat[i], b_flat[i])[0,1]
    return corr

def shuffle_inds(n):
    # Sample non-self indices. Used to create e.g. spatial correlation baseline 
    inds = []
    for i in range(n):
        possible = np.arange(n)
        possible = possible[possible != i]
        inds.append(np.random.choice(possible, replace=False))
    return inds