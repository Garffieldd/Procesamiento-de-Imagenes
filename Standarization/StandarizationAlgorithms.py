def intensity_rescaling(image):
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