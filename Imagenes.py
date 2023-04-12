from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

canvas = None

def open_image_umbralization():
    global canvas
    if isinstance(canvas, FigureCanvasTkAgg):
        canvas.get_tk_widget().destroy()
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    image_data = image_nibabel.get_fdata()
    tol=100
    tau=150
    while True:
        segmentation = image_data >= tau
        mBG = image_data[ segmentation == 0].mean()
        mFG = image_data[ segmentation == 1].mean()

        tau_post= 0.5 * (mBG + mFG)

        if np.abs(tau - tau_post) < tol:
            break
        else:
            tau = tau_post  
    
     
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.imshow(segmentation[:,:,20])
    ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()

#Por ahora solo lo hace en 3 separaciones, toca hacerlo dinamico
def open_image_Kmeans():
    global canvas
    if isinstance(canvas, FigureCanvasTkAgg):
        canvas.get_tk_widget().destroy()
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    image_data = image_nibabel.get_fdata()
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
        
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(segmentation[:,:,20])
    ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()
    global image_route
    image_route = route

#El codigo de regio Growing se demora mucho, toca mejorarlo
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

        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.imshow(segmentation[:,:,20])
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack()
        global image_route
        image_route = route   

   


root=Tk()
root.title("Cargar Imagen")
root.config(bg="black")
root.geometry("650x350")

buttonUmbra = Button(root, text="Select an image (umbralization)",command=open_image_umbralization)
buttonUmbra.pack()

buttonKmeans = Button(root, text="Select an image (K-means)",command=open_image_Kmeans)
buttonKmeans.pack()

#buttonRegionGrowing = Button(root, text="Select an image (regionGrowing)",command=open_image_regionGrowing)
#buttonRegionGrowing.pack()

image_route = None

root.mainloop()