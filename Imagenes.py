from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import threading


def open_image_umbralization():
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    image_data = image_nibabel.get_fdata()
    axial_slice = image_data[:, :, 20]
    tol=100
    tau=150
    while True:
        segmentation = axial_slice >= tau
        mBG = axial_slice[ segmentation == 0].mean()
        mFG = axial_slice[ segmentation == 1].mean()

        tau_post= 0.5 * (mBG + mFG)

        if np.abs(tau - tau_post) < tol:
            break
        else:
            tau = tau_post  
    
     
    segmentation_type = (segmentation * 255).astype(np.uint8) 
    image = Image.fromarray(segmentation_type)
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk
    global image_route
    image_route = route

#Por ahora solo lo hace en 3 separaciones, toca hacerlo dinamico
def open_image_Kmeans():
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    image_data = image_nibabel.get_fdata()
    axial_slice = image_data[:, :, 20]
    k1 = np.amin(axial_slice)
    k2 = np.mean(axial_slice)
    k3 = np.amax(axial_slice)

    for i in range(0,3):
        d1 = np.abs(k1 - axial_slice)  #diferencia entre la axial_slicen y el valor de referencia de cada cluster
        d2 = np.abs(k2 - axial_slice)
        d3 = np.abs(k3 - axial_slice)

        segmentation = np.zeros_like(axial_slice)
        segmentation[np.multiply(d1 < d2, d1 < d3)] = 0
        segmentation[np.multiply(d2 < d1, d2 < d3)] = 1
        segmentation[np.multiply(d3 < d1, d3 < d2)] = 2

        k1 = axial_slice[ segmentation == 0].mean()
        k2 = axial_slice[ segmentation == 1].mean()
        k3 = axial_slice[ segmentation == 2].mean() 
    
    
    segmentation_type = (segmentation * 255).astype(np.uint8) 
    image = Image.fromarray(segmentation_type)
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk
    global image_route
    image_route = route

def open_image_regionGrowing():
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    image_data = image_nibabel.get_fdata()
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

        axial_slice = segmentation[:,:,20]
        segmentation_type = (axial_slice * 255).astype(np.uint8) 
        image = Image.fromarray(segmentation_type)
        image_tk = ImageTk.PhotoImage(image)
        image_label.config(image=image_tk)
        image_label.image = image_tk
        global image_route
        image_route = route   

   


root=Tk()
root.title("Cargar Imagen")
root.config(bg="black")

"""
segmentationFrame = Frame()
segmentationFrame.pack()
segmentationFrame.config(bg="blue")
segmentationFrame.config(width="650", height="350")
"""

buttonUmbra = Button(root, text="Select an image (umbralization)",command=open_image_umbralization)
buttonUmbra.pack()

buttonKmeans = Button(root, text="Select an image (K-means)",command=open_image_Kmeans)
buttonKmeans.pack()

#buttonRegioGrowing = Button(root, text="Select an image (regionGrowing)",command=open_image_regionGrowing)
#buttonRegioGrowing.pack()

image_label = Label(root)
image_label.pack()

image_route = None

root.mainloop()