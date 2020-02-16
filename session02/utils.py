import numpy as np
from scipy.ndimage import measurements
    
def find_threshold(X, k):
    """Calculate empirical amplitude threshold.
    
    Parameters
    ----------
    X : array_like, shape (n_times,)
        Raw data trace.
    k : int
        Amplitude scalar.
        
    Returns
    -------
    sigma : float
        Amplitude threshold
        
    References
    -----
    [1] Rey, H. G., Pedreira, C., & Quiroga, R. Q. (2015). Past, present and future of 
        spike sorting techniques. Brain research bulletin, 119, 106-117.
    """
    return k * np.median( np.abs(X) ) / 0.6745

def peak_finder(X, thresh):
    """Simple peak finding algorithm.
    
    Parameters
    ----------
    X : array_like, shape (n_times,)
        Raw data trace.
    thresh : float
        Amplitude threshold.
        
    Returns
    -------
    peak_loc : array_like, shape (n_clusters,)
        Index of peak amplitudes.
    peak_mag : array_like, shape (n_clusters,)
        Magnitude of peak amplitudes.
    """
    
    ## Error-catching.
    assert X.ndim == 1
    
    ## Identify clusters.
    clusters, ix = measurements.label(X > thresh)
    
    ## Identify index of peak amplitudes. 
    peak_loc = np.concatenate(measurements.maximum_position(X, labels=clusters, index=np.arange(ix)+1))
    
    ## Identify magnitude of peak amplitudes.
    peak_mag = measurements.maximum(X, labels=clusters, index=np.arange(ix)+1)
    return peak_loc, peak_mag