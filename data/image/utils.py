import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt



def euclidean_dist(x1, x2):
    """
    Compute the euclidean distance between two points

    Parameters
    ----------
    x1 : list
        Coordinates of the first point
    x2 : list
        Coordinates of the second point

    Returns
    -------
    float
        Euclidean distance between the two points
    """
    return(np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2))

def phi(x,mu,sigma):
    """
    Compute the probability of a point being tumorous given its distance to the tumor edge. The probability is computed using a cumulative distribution function of a gaussian distribution and is normalized by the maximum value of the CDF.

    Parameters
    ----------
    x : float
        Distance to the tumor edge in mm
    mu : float
        Mean of the gaussian distribution in mm
    sigma : float
        Standard deviation of the gaussian distribution in mm

    Returns
    -------
    float
        Probability of the point being tumorous

    """
    #'Cumulative distribution function for the standard normal distribution'
    return (1-(erf((x-mu)/sigma / np.sqrt(2.0))))

def compute_distance_to_contour_3Dmap(imgsize,mask):
    """
    Compute the distance of each point of the image to the closest point of the mask

    Parameters
    ----------
    imgsize : tuple
        Dimensions of the image
    mask : np.array
        Binary mask of the contour

    Returns
    -------
    np.array
        3D array containing the distance of each point to the closest point of the mask
    """
    x = np.arange(0,imgsize[0], 1)
    y = np.arange(0, imgsize[1] , 1)
    z = np.arange(0, imgsize[2], 1)
    X, Y, Z = np.meshgrid(x, y, z,sparse=True)
    distance = np.zeros(imgsize)
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            for k in range(imgsize[2]):
                if mask[i,j,k] == 1:
                    distance[i,j,k] = 0
                else:
                    distance[i,j,k] = np.min([euclidean_dist([i,j,k],[x,y,z]) for x in range(imgsize[0]) for y in range(imgsize[1]) for z in range(imgsize[2]) if mask[x,y,z] == 1])
    return distance



