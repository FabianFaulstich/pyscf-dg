import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

from scipy.spatial import ConvexHull

def mol_size(Mol):

    expan = lambda cords: max(cords) - min(cords)
    minim = lambda cords: min(cords)
    out   = []
    m     =[]

    for i in range(3):
        out.append(expan([mol[i] for mol in Mol]))
        m.append(minim([mol[i] for mol in Mol]))

    return np.array(out), m

def RandomAtoms():
    xx = np.linspace(0, 5, 11)
    yy = np.linspace(0, 5, 11)
    zz = np.linspace(0, 5, 11)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms   = np.random.randint(0,5,(6,3)) 
    #np.array([[1,1,1], [4,4,1]])#, [4,1,1]])
    #print(atoms)
    #exit()

    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 5, 5, 5)
    print('Done! Elapsed time: ', time.time() - start)
    
    cmap = dg.get_cmap(len(atoms) + 1)
    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 

    for atom in atoms:
        ax.scatter3D(atom[0], atom[1], atom[2], color = "k", marker = "D");
    for i, col in enumerate(idx_mat.transpose()):
        for k, point in enumerate(col):
            if point:
                ax.scatter3D(grid[k][0], grid[k][1], grid[k][2], 
                        color = cmap(i), marker = "+");            
    plt.show()


def CH4():

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams.update({'font.size': 14.5})


    xx = np.linspace(0, 18, 131) #131
    yy = np.linspace(0, 18, 131)
    zz = np.linspace(0, 18, 131)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])

    atoms = [[ 0, 0, 0], [ 1.186, 1.186, 1.186],
             [ 1.186,-1.186,-1.186],[-1.186, 1.186,-1.186],
             [-1.186,-1.186, 1.186]]

    # Centering Molecule in Box:
    ms, mm = mol_size(atoms)
    offset = np.array([ (bs - s)/2. - m for bs, s, m in zip([18]*3, ms, mm)])

    for k, off in enumerate(offset):
        for j in range(len(atoms)):
            atoms[j][k] += off

    
    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 18, 18, 18)
    print('Done! Elapsed time: ', time.time() - start)

    cmap = dg.get_cmap(len(atoms) + 1)
    fig, _ = plt.subplots(nrows=1, ncols=1,  figsize=(7,7))
    ax = plt.axes(projection ="3d")

    #for i, col in enumerate(idx_mat.transpose()):
    #    if i == 0:
    #        for k, point in enumerate(col):
    #            if point:
    #                ax.scatter3D(grid[k][0], grid[k][1], grid[k][2],
    #                        cmap='viridis', marker = ".");
    
    vcell = idx_mat[:,1]
    points = grid[vcell,:]
    points = points[points[:,0] <= 11]
    points = points[7 <= points[:,0]]
    points = points[points[:,1] <= 11]
    points = points[7 <= points[:,1]]
    points = points[points[:,2] <= 11]
    points = points[7 <= points[:,2]]
    
    ax.scatter3D(points[:,0], points[:,1], points[:,2], 
            #color ='lightsalmon', edgecolors='salmon'); 
            cmap ='hsv', edgecolors='none'); 

    vcell = idx_mat[:,4]
    points = grid[vcell,:]
    points = points[points[:,0] <= 11]
    points = points[7 <= points[:,0]]
    points = points[points[:,1] <= 11]
    points = points[7 <= points[:,1]]
    points = points[points[:,2] <= 11]
    points = points[7 <= points[:,2]]
    
    ax.scatter3D(points[:,0], points[:,1], points[:,2], 
            cmap ='hsv', edgecolors='none');
    
    
    vcell = idx_mat[:,0]
    points = grid[vcell,:]
    hullpoints = ConvexHull(points)
    for simplex in hullpoints.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 
                'k-', linewidth = 2)
   

    edges = points[hullpoints.vertices,:]
    ax.plot([edges[0][0], edges[-1][0]],
            [edges[0][1], edges[-1][1]], 
            [edges[0][2], edges[-1][2]], 'k-', linewidth = 2)
    for edge in edges:
        ax.plot([edge[0]], [edge[1]], [6], color = "k",
                marker = "D", markersize=5, mew=.5)
        ax.plot([edge[0], edge[0]],[edge[1], edge[1]], [edge[2], 6],
                    color = 'k', linestyle = ':', linewidth=1)

    ax.scatter3D(points[:,0], points[:,1], points[:,2], 
            #color ='lightsteelblue', edgecolors='cornflowerblue');
            cmap ='hsc', edgecolors='none');


    #for simplex in hullpoints.simplices:
    #    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 
    #            'k-', linewidth = 2)
    #    ax.plot(points[simplex, 0], points[simplex, 1], [6.5], 'k-',
    #            linewidth = 2)
    # Projection
    for i, atom in enumerate(atoms):
        i#ax.scatter([atom[0]], [atom[1]], [6.5],s=250, facecolors='none', 
         #        edgecolors='k')

        if i == 0:
            ax.plot([atom[0]], [atom[1]], [6], color = "dimgray",
                     marker = ".", markersize=15, mew=3)
        else:
            ax.plot([atom[0]], [atom[1]], [6], color = "royalblue", 
                    marker = ".", markersize=15, mew=3)


    for i, atom in enumerate(atoms):
        if i == 0:
            ax.plot([atom[0]], [atom[1]], [atom[2]], color = "k",
                    marker = ".", markersize=50)
            ax.plot([atom[0], atom[0]],[atom[1], atom[1]], [atom[2], 6], 
                    color = 'k', linestyle = ':', linewidth=1.5)
        else:
            ax.plot([atom[0],atoms[0][0]],[atom[1],atoms[0][1]],
                    [atom[2],atoms[0][2]], color = 'k', linewidth=4)
            ax.plot([atom[0]], [atom[1]], [atom[2]], color = "b", 
                    marker = ".", markersize=40)
            ax.plot([atom[0], atom[0]],[atom[1], atom[1]], [atom[2], 6], 
                    color = 'k', linestyle = ':', linewidth=1.5)
    
    #ax.set_xlabel('X-axis') 
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Z-axis')
    
    ax.set_xlim((7,11))
    ax.set_ylim((7,11)) 
    ax.set_zlim((6,11)) 

    ax.xaxis.set_ticks([7,8,9,10,11])
    ax.yaxis.set_ticks([7,8,9,10,11])
    ax.zaxis.set_ticks([6,7,8,9,10,11])
    
    # For generating clip
    #for i in np.arange(1080):
    #    ax.view_init(elev=7., azim= i)
    #    fig.subplots_adjust(top=1.0, bottom = .04, left = 0.0, right = .97)
    #    fig.savefig("movie-" + "{0:0=4d}".format(i) + ".png")
    ax.view_init(elev=7., azim= -36)
    plt.show()

def timing():

    xx = np.linspace(0, 12, 25)
    yy = np.linspace(0, 12, 25)
    zz = np.linspace(0, 12, 25)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms   = np.array([[5, 5, 6], [7, 5, 6], [5, 7, 6], [7, 7, 6]]) 
      
    
    print("Starting naive Voronoi:")
    vstart = time.time()
    idx_mat = np.zeros((grid.shape[0],len(atoms)), dtype = bool)
    for j, point in enumerate(grid):
        k = dg.get_dist_atom(atoms, 12, 12, point)
        idx_mat[j,k] = True
    vend = time.time()
    print(idx_mat.shape)
    print("Computational time for naive voronoi: ", vend - vstart)
    
    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 12, 12, 12)
    print('Done! Elapsed time: ', time.time() - start)


def graphene3d():

    xx = np.linspace(0, 2.45, 50)
    yy = np.linspace(0, 4.2435244, 50)
    zz = np.linspace(0, 10, 1)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms = [[0., 0., 0.], [0., 1.4145081595145836, 0.],
             [1.225, 2.122, 0.], [1.225,3.5362703987864585, 0.]]

    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 2.45, 4.2435244, 10)
    print('Done! Elapsed time: ', time.time() - start)
    
    #cmap = dg.get_cmap(len(atoms) + 1)
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
   
    for i in range(4):
        vcell = idx_mat[:,i]
        points = grid[vcell,:]

        ax.scatter3D(points[:,0], points[:,1], points[:,2],
            #color ='lightsteelblue', edgecolors='cornflowerblue');
            cmap ='hsc', edgecolors='none');

    for atom in atoms:
        ax.plot([atom[0]], [atom[1]], [atom[2]], color = "b",
                marker = ".", markersize=40)
    plt.show()

def graphene2d_small():

    xx = np.linspace(0, 2.45, 50)
    yy = np.linspace(0, 4.2435244, 50)
    zz = np.linspace(0, 10, 1)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms = [[0., 0., 0.], [0., 1.4145081595145836, 0.],
             [1.225, 2.122, 0.], [1.225,3.5362703987864585, 0.]]

    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 2.45, 4.2435244, 10)
    print('Done! Elapsed time: ', time.time() - start)
    
    cmap = dg.get_cmap(len(atoms) + 1)
    fig = plt.figure(figsize = (10, 7))
  
    #for i, col in enumerate(mat.transpose()):
    #    for k, point in enumerate(col):
    #        if point and coords[k][2] == sl:
    #            plt.plot(coords[k][0],coords[k][1], color =cmap(i) , marker='x')

    for i in range(4):
        vcell = idx_mat[:,i]
        points = grid[vcell,:]
        plt.scatter(points[:,0], points[:,1], cmap ='hsc', edgecolors='none');
        #for point in points: 
        #    plt.plot(point[0], point[1], color =cmap(i), marker = ".", 
        #            markersize=15)
            #color ='lightsteelblue', edgecolors='cornflowerblue');
            #color =cmap(i));

    for atom in atoms:
        plt.plot([atom[0]], [atom[1]], color = "k",
                marker = ".", markersize=40)
    plt.show()

def graphene(size):

    
    
    if size == '4':
        # small:
        xx = np.linspace(0, 2.45, 70)
        yy = np.linspace(0, 4.2435, 70)
        zz = np.linspace(0, 10, 1)
    
        grid = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    
        atoms = [[0., 0., 0.], [0., 1.415, 0.],
                 [1.225, 2.122, 0.], [1.225,3.536, 0.]]
    
        idx_mat = dg.naive_voronoi(grid, atoms, 2.45, 4.2435, 10)

    elif size == '8':
        xx = np.linspace(0, 4.9, 70)
        yy = np.linspace(0, 4.2435, 70)
        zz = np.linspace(0, 10, 1)

        grid = np.array([[x,y,z] for x in xx for y in yy for z in zz])

        atoms = [[0.0, 0.0, 0.0],
                 [0.0, 1.415, 0.0],
                 [1.225, 2.122, 0.0],
                 [1.225, 3.536, 0.0],
                 [2.450, 0.0, 0.0],
                 [2.450, 1.415, 0.0],
                 [3.675, 2.122, 0.0],
                 [3.675, 3.536, 0.0]]

        idx_mat = dg.naive_voronoi(grid, atoms, 4.9, 4.2435, 10)
    
    elif size == '16':
        # large:
        xx = np.linspace(0, 4.9, 70)
        yy = np.linspace(0, 8.4870489, 70)
        zz = np.linspace(0, 10, 1)

        grid = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    
        atoms = [[0., 0., 0.], [0., 1.415, 0.],
                 [1.225, 2.122, 0.], [1.225,3.536, 0.],
                 [0.,4.244, 0.], [0., 5.658, 0.], [1.225, 6.365, 0.],
                 [1.225, 7.780, 0.], [2.45, 0.0, 0.0], [2.450, 1.415, 0.],
                 [3.675, 2.123, 0.], [3.675, 3.536, 0.], [2.45, 4.244, 0.],
                 [2.450, 5.658, 0.], [3.675, 6.365, 0.], [3.675, 7.7880, 0.]]
    
        idx_mat = dg.naive_voronoi(grid, atoms, 4.9, 8.4870489, 10)
            
    elif size == '24':
        # 24 atoms:
        xx = np.linspace(0, 7.35, 70)
        yy = np.linspace(0, 8.4870489, 70)
        zz = np.linspace(0, 10, 1)
    
        grid = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    
        atoms = [[0., 0., 0.], [0., 1.415, 0.],
                 [1.225, 2.122, 0.], [1.225,3.536, 0.],
                 [0.,4.244, 0.], [0., 5.658, 0.], [1.225, 6.365, 0.],
                 [1.225, 7.780, 0.], [2.45, 0.0, 0.0], [2.450, 1.415, 0.],
                 [3.675, 2.123, 0.], [3.675, 3.536, 0.], [2.45, 4.244, 0.],
                 [2.450, 5.658, 0.], [3.675, 6.365, 0.], [3.675, 7.7880, 0.],
                 [4.9, 0.0, 0.0],
                 [4.9, 1.415, 0.0],
                 [6.125, 2.122, 0.0],
                 [6.125, 3.536, 0.0],
                 [4.9, 4.244, 0.0],
                 [4.9, 5.658, 0.0],
                 [6.125, 6.365, 0.0],
                 [6.125, 7.780, 0.0]
                 ]
    
        idx_mat = dg.naive_voronoi(grid, atoms, 7.350, 8.4870489, 10)




    c_map = dg.get_cmap(len(atoms)+2, 'nipy_spectral')
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize = (6, 8))

    #for i, col in enumerate(mat.transpose()):
    #    for k, point in enumerate(col):
    #        if point and coords[k][2] == sl:
    #            plt.plot(coords[k][0],coords[k][1], color =cmap(i) , marker='x')

    for i in range(len(atoms)): 
        vcell = idx_mat[:,i]
        points = grid[vcell,:]
        plt.scatter(points[:,0], points[:,1], color = c_map(i), edgecolors='none');
        #for point in points: 
        #    plt.plot(point[0], point[1], color =cmap(i), marker = ".", 
        #            markersize=15)
            #color ='lightsteelblue', edgecolors='cornflowerblue');
            #color =cmap(i));

    for atom in atoms:
        plt.plot([atom[0]], [atom[1]], color = "k",
                marker = ".", markersize=40)
    
    plt.show()




if __name__ == '__main__':

    CH4()
    #graphene('4')
    #graphene('8')
    #graphene('16')
    #graphene('24')   
