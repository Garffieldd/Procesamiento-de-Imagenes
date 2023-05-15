import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def mean_filter(image_data):
    print ( "entre al mean filter ")
    filtered_image_data = np.zeros_like(image_data)
    for x in range(0, image_data.shape[0]-2) :
        for y in range(0, image_data.shape[1]-2) :
            for z in range(0, image_data.shape[2]-2) :
                avg = 0
                for dx in range(-1, 1) :
                    for dy in range(-1, 1) :
                        for dz in range(-1, 1) :
                            avg = avg + image_data[x+dx, y+dy, z+dz]

                filtered_image_data[x+1, y+1, z+1] = avg / 27

    return filtered_image_data
            
def median_filter(image_data):
    filtered_image = np.zeros_like(image_data)

    for x in range(0, image_data.shape[0]-1):
        for y in range(0, image_data.shape[1]-1):
            for z in range(0, image_data.shape[2]-1):
                # Extraer la vecindad 3x3x3
                neighborhood = image_data[x-1:x+2, y-1:y+2, z-1:z+2]
                
                # Ordenar la vecindad en orden ascendente
                sorted_neighborhood = np.sort(neighborhood, axis=None)

                # Calcular la mediana de la vecindad
                median_value = np.median(sorted_neighborhood)
                
                # Asignar el valor mediano al píxel filtrado
                filtered_image[x, y, z] = median_value

    return filtered_image