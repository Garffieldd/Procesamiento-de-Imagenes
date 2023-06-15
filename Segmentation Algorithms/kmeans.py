import numpy as np


def kmeans_segmentation(image, k):
    
    centroids = np.linspace(np.amin(image), np.amax(image), k)
    iteration = 15
    for i in range(iteration):
        distances = [np.abs(k - image) for k in centroids] #diferencia entre la image_data y el valor de referencia de cada cluster
        segmentation = np.argmin(distances, axis=0)
        for group in range(k):
            centroids[group] = image[segmentation == group].mean()
    
    return segmentation