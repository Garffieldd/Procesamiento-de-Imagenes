import numpy as np

def regios_growing_segmentation(image):
    image_data = image.get_fdata()
    tol = 3
    segmentation = np.zeros_like(image_data)

# Recorrer todos los voxels de la image_datan
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            for z in range(image_data.shape[2]):
                if segmentation[x, y, z] == 0:  # Si el voxel no ha sido segmentado previamente
                    valor_medio_cluster = image_data[x, y, z]
                    queue = [(x, y, z)]  # Cola para realizar el crecimiento de la región

                    # Aplicar el algoritmo de region growing usando una cola
                    while queue:
                        voxel = queue.pop(0)
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    nx = voxel[0] + dx
                                    ny = voxel[1] + dy
                                    nz = voxel[2] + dz
                                    if (0 <= nx < image_data.shape[0] and 0 <= ny < image_data.shape[1] and
                                        0 <= nz < image_data.shape[2] and np.abs(valor_medio_cluster - image_data[nx, ny, nz]) < tol):
                                        segmentation[nx, ny, nz] = 1
                                        queue.append((nx, ny, nz))
                                    else:
                                        segmentation[nx, ny, nz] = 0
                                        queue.append((nx, ny, nz))
                    # Calcular el nuevo valor medio del cluster
                    cluster_voxels = image_data[segmentation == 1]
                    valor_medio_cluster = cluster_voxels.mean()
    return segmentation