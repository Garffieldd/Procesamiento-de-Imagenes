import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.signal import find_peaks
from scipy import stats as st
import statistics as stat

def intensity_rescaling(image):
    print("entre")
    image_data = image.get_fdata()
    min_value = image_data.min()

    max_value = image_data.max()

    image_data_rescaled = (image_data - min_value) / (max_value - min_value)
    return image_data_rescaled


def z_score_tranformation(image):
    image_data = image.get_fdata()
    media = image_data[image_data > 10].mean()  # hay que excluir el fondo de los resultados por eso el image_data > 10
    desviacion_estandar = image_data[image_data > 10].std()

    image_data_Z_SCORE = (image_data -  media) / desviacion_estandar
    return image_data_Z_SCORE

# def histogram_matching(imgOrigin,imgTarget):
#     #histogram
#     data_orig = imgOrigin.get_fdata()
#     data_target = imgTarget.get_fdata()

#     # Redimensionar los datos en un solo arreglo 1D
#     flat_orig = data_orig.flatten()
#     flat_target = data_target.flatten()

#     # Calcular los histogramas acumulativos
#     hist_orig, bins = np.histogram(flat_orig, bins=256, range=(0, 255), density=True)
#     hist_orig_cumulative = hist_orig.cumsum()
#     hist_target, _ = np.histogram(flat_target, bins=256, range=(0, 255), density=True)
#     hist_target_cumulative = hist_target.cumsum()

#     # Mapear los valores de la imagen de origen a los valores de la imagen objetivo
#     lut = np.interp(hist_orig_cumulative, hist_target_cumulative, bins[:-1])

#     # Aplicar el mapeo a los datos de la imagen de origen
#     data_matched = np.interp(data_orig, bins[:-1], lut)

#     # Crear una nueva imagen con los datos estandarizados
#     image_matched = nib.Nifti1Image(data_matched, imgOrigin.affine,imgOrigin.header)
#     data = image_matched.get_fdata()
#     return data

def histogram_matching(objective_data, origin_data,k):
    # Reshape the data arrays to 1D arrays
    objective_flat = objective_data.get_fdata().flatten()
    origin_flat = origin_data.get_fdata().flatten()


    reference_landmarks = np.percentile(objective_flat, np.linspace(0, 100, k))
    transform_landmarks = np.percentile(origin_flat, np.linspace(0, 100, k))

    piecewise_func = np.interp(origin_flat, transform_landmarks, reference_landmarks)


    transformed_data = piecewise_func.reshape(origin_data.shape)

    return transformed_data

def white_stripe(image):
    image_data = image.get_fdata()
    # Calcular el histograma
    hist, bin_edges = np.histogram(image_data.flatten(), bins=100)
    # Encontrar los picos del histograma
    picos, _ = find_peaks(hist, height=100)
    val_picos=bin_edges[picos]

    # Imagen reecalada
    image_data_rescaled=image_data/val_picos[1]

    return image_data_rescaled
    # # Mostrar el histograma con los picos identificados
    # plt.axvline(val_picos[0], color='r', linestyle='--')
    # plt.hist(image_data.flatten(), bins=100)
    # plt.plot(bin_edges[picos], hist[picos], "x")
    # plt.show()
    # plt.hist(image_data_rescaled.flatten(), 100)