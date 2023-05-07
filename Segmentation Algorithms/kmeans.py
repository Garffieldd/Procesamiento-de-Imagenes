import numpy as np

def kmeans_segmentation(image):
    image_data = image.get_fdata()
        #axial_slice = image_data[:, :, 20]
    k1 = np.amin(image_data)
    k2 = np.mean(image_data)
    k3 = np.amax(image_data)

    for i in range(0,3):
        d1 = np.abs(k1 - image_data)  #diferencia entre la image_datan y el valor de referencia de cada cluster
        d2 = np.abs(k2 - image_data)
        d3 = np.abs(k3 - image_data)

        segmentation = np.zeros_like(image_data)
        segmentation[np.multiply(d1 < d2, d1 < d3)] = 0
        segmentation[np.multiply(d2 < d1, d2 < d3)] = 1
        segmentation[np.multiply(d3 < d1, d3 < d2)] = 2

        k1 = image_data[ segmentation == 0].mean()
        k2 = image_data[ segmentation == 1].mean()
        k3 = image_data[ segmentation == 2].mean() 
    
    return segmentation