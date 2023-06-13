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

    for x in range(1, image_data.shape[0]-1):
        for y in range(1, image_data.shape[1]-1):
            for z in range(1, image_data.shape[2]-1):
                # Extraer la vecindad 3x3x3
                neighborhood = image_data[x-1:x+2, y-1:y+2, z-1:z+2]
                # Calcular la mediana de la vecindad
                median_value = np.median(neighborhood)
                
                # Asignar el valor mediano al píxel filtrado
                filtered_image[x, y, z] = median_value

    return filtered_image


def median_filter_with_borders(image_data):
        threshold = 2500

        filtered_image = np.zeros_like(image_data)

        for x in range(1, image_data.shape[0] - 2):
            for y in range(1, image_data.shape[1] - 2):
                for z in range(1, image_data.shape[2] - 2):
                    # Compute the derivatives in x, y, and z directions
                    dx = image_data[x + 1, y, z] - image_data[x - 1, y, z]
                    dy = image_data[x, y + 1, z] - image_data[x, y - 1, z]
                    dz = image_data[x, y, z + 1] - image_data[x, y, z - 1]

                    # Compute the magnitude of the gradient
                    magnitude = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # Separate pixels based on the current threshold
                    below_threshold = magnitude[magnitude < threshold]
                    above_threshold = magnitude[magnitude >= threshold]

                    if len(below_threshold) > 0 and len(above_threshold) > 0:
                        threshold = (np.mean(below_threshold) + np.mean(above_threshold)) / 2
                    else:
                        threshold = 0  


                    # If the magnitude is below the threshold, apply median filter
                    if magnitude < threshold:
                        for x in range(1, image_data.shape[0]-1):
                            for y in range(1, image_data.shape[1]-1):
                                for z in range(1, image_data.shape[2]-1):
                                    # Extraer la vecindad 3x3x3
                                    neighborhood = image_data[x-1:x+2, y-1:y+2, z-1:z+2]
                                    # Calcular la mediana de la vecindad
                                    median_value = np.median(neighborhood)
                                    
                                    # Asignar el valor mediano al píxel filtrado
                                    filtered_image[x, y, z] = median_value
                        else:
                            filtered_image[x, y, z] = image_data[x, y, z]

        return filtered_image



def mean_filter_with_borders(image_data):
        print("Entre al mean_filter_with_borders")
        threshold = 2500

        filtered_image_data = np.zeros_like(image_data)

        for x in range(1, image_data.shape[0] - 2):
            for y in range(1, image_data.shape[1] - 2):
                for z in range(1, image_data.shape[2] - 2):
                    # Compute the derivatives in x, y, and z directions
                    dx = image_data[x + 1, y, z] - image_data[x - 1, y, z]
                    dy = image_data[x, y + 1, z] - image_data[x, y - 1, z]
                    dz = image_data[x, y, z + 1] - image_data[x, y, z - 1]

                    # Compute the magnitude of the gradient
                    magnitude = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # Separate pixels based on the current threshold
                    below_threshold = magnitude[magnitude < threshold]
                    above_threshold = magnitude[magnitude >= threshold]

                    if len(below_threshold) > 0 and len(above_threshold) > 0:
                        threshold = (np.mean(below_threshold) + np.mean(above_threshold)) / 2
                    else:
                        threshold = 0  


                    # If the magnitude is below the threshold, apply median filter
                    if magnitude < threshold:
                        for x in range(0, image_data.shape[0]-2) :
                            for y in range(0, image_data.shape[1]-2) :
                                for z in range(0, image_data.shape[2]-2) :
                                    avg = 0
                                    for dx in range(-1, 1) :
                                        for dy in range(-1, 1) :
                                            for dz in range(-1, 1) :
                                                avg = avg + image_data[x+dx, y+dy, z+dz]

                                    filtered_image_data[x+1, y+1, z+1] = avg / 27
                    else:
                        filtered_image_data[x, y, z] = image_data[x, y, z]
        return filtered_image_data