import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KDTree

cdict = {'red': [(0.0,  0.0, 0.0),
                 (0.33, 0.0, 0.0),
                 (0.66,  1.0, 1.0),
                 (1.0,  1.0, 1.0)],
         'blue': [(0.0,  0.0, 0.0),
                  (0.33, 1.0, 1.0),
                  (0.66,  0.0, 0.0),
                  (1.0,  0.0, 0.0)],
         'green': [(0.0,  0.0, 0.0),
                   (0.33, 0.0, 0.0),
                   (0.66,  0.0, 0.0),
                   (1.0,  1.0, 1.0)]}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

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

def plot_cvt(ax, centroids, fit, desc, x,dim1,dim2):
    # compute Voronoi tesselation
    print("Voronoi...")
    vor = Voronoi(centroids[:,0:2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    norm = matplotlib.colors.Normalize(vmin=min(fit), vmax=max(fit))
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
            print(k)
    fit_reshaped = fit.reshape((len(fit),))
    sc = ax.scatter(desc[:,0], desc[:,1], c=fit_reshaped, cmap=my_cmap, s=10, zorder=0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: %s centroids_file archive.dat' % sys.argv[0])

    centroids = load_centroids(sys.argv[1])
    dim_x = 24
    fit, beh, x = load_data(sys.argv[2], centroids.shape[1],dim_x)
    print("Fitness max : ", max(fit))
    index = np.argmax(fit)
    print("Associated desc : " , beh[index] )
    print("Associated ctrl : " , x[index] )
    print("Index : ", index)
    print("total len ",len(fit))

    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    plot_cvt(axes, centroids, fit, beh, x,2,4)
    fig.savefig('cvt.pdf')
    fig.savefig('cvt.png')
