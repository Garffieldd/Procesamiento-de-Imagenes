import numpy as np


def region_growing_segmentation(image, tol):
    segmentation = np.zeros_like(image)
    x = 120
    y = 120
    z = 44
    if segmentation[x,y,z] == 1:
        return
    valor_medio_cluster = image[x,y,z]
    segmentation[x,y,z] = 1
    vecinos = [(x, y, z)]
    while vecinos:
        x, y, z = vecinos.pop()
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx >= 0 and nx < image.shape[0] and \
                        ny >= 0 and ny < image.shape[1] and \
                        nz >= 0 and nz < image.shape[2]:
                        if np.abs(valor_medio_cluster - image[nx,ny,nz]) < tol and \
                            segmentation[nx,ny,nz] == 0:
                            segmentation[nx,ny,nz] = 1
                            vecinos.append((nx, ny, nz))
    return segmentation