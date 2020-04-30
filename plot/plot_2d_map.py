#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KDTree
import matplotlib.cm as cm

my_cmap = cm.viridis

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def load_data(filename, dim,dim_x):
    print("Loading ",filename)
    data = np.loadtxt(filename)
    fit = data[:, 0:1]
    desc = data[:,1: dim+1]
    x = data[:,dim+1:dim+1+dim_x]

    return fit, desc, x

def load_centroids(filename):
    points = np.loadtxt(filename)
    return points

def plot_cvt(ax, centroids, fit, desc, x,dim1,dim2, min_fit, max_fit):
    # compute Voronoi tesselation
    print("Voronoi...")
    vor = Voronoi(centroids[:,0:2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    print("fit:", min_fit, max_fit)
    norm = matplotlib.colors.Normalize(vmin=min_fit, vmax=max_fit)
    print("KD-Tree...")
    kdt = KDTree(centroids, leaf_size = 30, metric = 'euclidean')

    print("plotting contour...")
    #ax.scatter(centroids[:, 0], centroids[:,1], c=fit)
    # contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor='black', facecolor='white', lw=1)

    print("plotting data...")
    k = 0
    for i in range(0, len(desc)):
        q = kdt.query([desc[i]], k = 1)
        index = q[1][0][0]
        region = regions[index]
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.9, color=my_cmap(norm(fit[i]))[0])
        k += 1
        if k % 100 == 0:
            print(k, end=" ", flush=True)
    fit_reshaped = fit.reshape((len(fit),))
    sc = ax.scatter(desc[:,0], desc[:,1], c=fit_reshaped, cmap=my_cmap, s=10, zorder=0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: %s centroids_file archive.dat [min_fit] [max_fit]' % sys.argv[0])

    centroids = load_centroids(sys.argv[1])
    dim_x = 24
    fit, beh, x = load_data(sys.argv[2], centroids.shape[1], dim_x)
    print("Fitness max : ", max(fit))
    index = np.argmax(fit)
    print("Average fit:", fit.sum() / fit.shape[0])
    print("Associated desc : " , beh[index] )
    print("Associated ctrl : " , x[index] )
    print("Index : ", index)
    print("total len ",len(fit))

    if len(sys.argv) > 3:
        min_fit = float(sys.argv[3])
        max_fit = float(sys.argv[4])
    else:
        min_fit = min(fit)
        max_fit = max(fit)
    print("Min = {} Max={}".format(min_fit, max_fit))
    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    plot_cvt(axes, centroids, fit, beh, x,2,4, min_fit, max_fit)
    fig.savefig('cvt.pdf')
    fig.savefig('cvt.png')
