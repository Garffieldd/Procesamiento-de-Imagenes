from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
sys.path.append('./Segmentation Algorithms')
sys.path.append('./Standarization')
from umbralization import umbralization_segmentation
from kmeans import kmeans_segmentation
from region_growing import region_growing_segmentation
from StandarizationAlgorithms import *

#canvas = None
image_data = None
scale_num = 0
selected_segmentation = None
preview = None
selected_image = None
fig = None


# Funcion que se llama cuando se carga la imagen
def recover_image():
    global selected_image
    global fig
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    selected_image = nib.load(route)

def combo_option_standarization(image):
    global preview
    global image_data
    selected_standarization = standarizationSelector.get()
    if(selected_standarization == "Intensity rescaling"):
        image_data = intensity_rescaling(image)
        preview = intensity_rescaling(image)
        create_preview(preview)
    elif(selected_standarization == "Z-score transformation"):
        image_data = z_score_tranformation(image)
        preview = z_score_tranformation(image)
        create_preview(preview)
    elif(selected_standarization == "White straping"):
        image_data = z_score_tranformation(image.get_fdata())
        preview = z_score_tranformation(image.get_fdata())
        create_preview(preview)

def create_preview(data):
    global scale_num
    figPre.clf()
    ax = figPre.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(data[:,:,scaleNum])
    ax.axis('off')
    canvaPre.draw()  



def combo_option_segmentation():
    global image_data
    global selected_segmentation
    selected_segmentation = segmentationSelector.get()
    if selected_segmentation is None:
        print("No hay data")
    elif(selected_segmentation == "Umbralizacion"):
        open_image_umbralization(image_data)
    elif(selected_segmentation == "K-means"):
        open_image_Kmeans(image_data)
    elif(selected_segmentation == "Region Growing"):
        open_image_Kmeans(image_data)

def scale_widget_option(value):
    global scale_num
    global selected_segmentation
    global fig

    
    if image_data is not None:
        scale_num = value

        if selected_segmentation == "Umbralizacion":
            open_image_umbralization(image_data)
            create_preview(preview)
        elif selected_segmentation == "K-means":
            open_image_Kmeans(image_data)
            create_preview(preview)
        elif selected_segmentation == "Region Growing":
            open_image_regionGrowing(image_data)
            create_preview(preview)

def open_image_umbralization(image):
    global scale_num
    segmentation = umbralization_segmentation(image) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw()  
    

#Por ahora solo lo hace en 3 separaciones, toca hacerlo dinamico
def open_image_Kmeans(image):
    global scale_num
    segmentation = kmeans_segmentation(image) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 
    

#El codigo de regio Growing se demora mucho, toca mejorarlo
def open_image_regionGrowing(image):
    global scale_num
    segmentation = region_growing_segmentation(image)
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 


def close_interface():
    plt.close(fig)
    plt.close(figPre)
    canvas.get_tk_widget().destroy()
    canvaPre.get_tk_widget().destroy()
    optionFrame.destroy()
    imageFrame.destroy()
    root.destroy()
        
#Creacion de componentes de la interfaz

root=Tk()
root.title("Cargar Imagen")
root.protocol("WM_DELETE_WINDOW", close_interface)


#---------------------- FRAME DE LAS OPCIONES
optionFrame = Frame(root)
optionFrame.grid(column=0, row=0)
optionFrame.config(width=300,height=600, bg='dark turquoise')
#------------------------ FRAME DE LA VISUALIZACION DE LA IMAGEN
imageFrame = Frame(root)
imageFrame.grid(column=1, row=0)
imageFrame.config(width=500,height=600, bg='turquoise')

#canvas de la imagen segmentada
fig = plt.figure(figsize=(6, 6), dpi=100)
fig.tight_layout(pad=0)
canvas = FigureCanvasTkAgg(fig, master=imageFrame)
canvas.get_tk_widget().pack()

#canvas de la preview de la imagen
figPre = plt.figure(figsize=(6, 6), dpi=100)
figPre.tight_layout(pad=0)
canvaPre = FigureCanvasTkAgg(figPre, master=optionFrame) 
canvaPre.get_tk_widget().grid(column = 0,row = 5,padx=10,pady=10) 
canvaPre.get_tk_widget().config(width=200,height=200)




for i in range(10):
    optionFrame.grid_rowconfigure(i, weight=1)
    optionFrame.grid_columnconfigure(i, weight=1)

buttonImageSelector = Button(optionFrame, text="Upload an image .nii.gz",command=recover_image)
buttonImageSelector.grid(column = 0, row = 0,padx=100,pady=10)

standarizationSelector = ttk.Combobox(optionFrame,values=["Intensity rescaling","Z-score transformation","White straping"])
standarizationSelector.grid(column = 0, row = 1, padx = 100, pady = 10)
standarizationSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_standarization(selected_image))

segmentationSelector = ttk.Combobox(optionFrame,values=["Umbralizacion","K-means","Region Growing"])
segmentationSelector.grid(column = 0, row = 2, padx = 100, pady = 10)

scaleWidget = Scale(optionFrame,from_=0,to=48,orient= HORIZONTAL, command=scale_widget_option)
scaleWidget.grid(column=0,row=3,padx=10,pady=100)

buttonSegmentate = Button(optionFrame, text="Segmentate",command=combo_option_segmentation)
buttonSegmentate.grid(column = 0, row = 4,padx=10,pady=100)


root.mainloop()