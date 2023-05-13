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
sys.path.append('./Noise Remotion')
from umbralization import umbralization_segmentation
from kmeans import kmeans_segmentation
from region_growing import region_growing_segmentation
from StandarizationAlgorithms import *
from NoiseRemotion import *

#canvas = None
image_data = None
scale_num = 0
selected_segmentation = None
preview = None
selected_image = None
selected_image_target = None
fig = None
tol = None
tau = None
k = None




# Funcion que se llama cuando se carga la imagen
def recover_image():
    global selected_image
    global fig
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    selected_image = nib.load(route)
    
def recover_image_target():
    global selected_image_target
    global fig
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    selected_image_target = nib.load(route)

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
    elif(selected_standarization == "Histogram matching"):
        # route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
        # selected_image_target = nib.load(route)
        image_data = histogram_matching(image,selected_image_target)
        preview = histogram_matching(image,selected_image_target)
        create_preview(preview)
    elif(selected_standarization == "White straping"):
        image_data = white_stripe(image)
        preview = white_stripe(image)

def combo_option_noise_remotion(data):
    global image_data
    global preview
    selected_sound_remotion = noiseRemotionSelector.get()
    if (selected_sound_remotion == "Mean filter"):
        image_data = mean_filter(data)
        preview = mean_filter(data)
        create_preview(preview)
    elif(selected_sound_remotion == "Medium filter"):
        image_data = mean_filter(data)
        preview = mean_filter(data)
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
    global tol
    global tau
    global k 
    selected_segmentation = segmentationSelector.get()
    tol = int(entryTol.get()) if entryTol.get() else None
    tau = int(entryTau.get()) if entryTau.get() else None
    k = int(entryK.get()) if entryK.get() else None
    if selected_segmentation is None:
        print("No hay data")
    elif(selected_segmentation == "Umbralizacion"):
        open_image_umbralization(image_data,tol,tau)
    elif(selected_segmentation == "K-means"):
        open_image_Kmeans(image_data,k)
    elif(selected_segmentation == "Region Growing"):
        open_image_regionGrowing(image_data,tol)

def scale_widget_option(value):
    global scale_num
    global selected_segmentation
    global fig
    global tol
    global tau
    global k
    
    if image_data is not None:
        scale_num = value

        if selected_segmentation == "Umbralizacion":
            open_image_umbralization(image_data,tol,tau)
            create_preview(preview)
        elif selected_segmentation == "K-means":
            open_image_Kmeans(image_data,k)
            create_preview(preview)
        elif selected_segmentation == "Region Growing":
            open_image_regionGrowing(image_data,tol)
            create_preview(preview)

def segmentation_params():
    selected_value = segmentationSelector.get()
    labelTol.grid_remove()
    labelTau.grid_remove()
    labelK.grid_remove()
    entryTol.grid_remove()
    entryTau.grid_remove()
    entryK.grid_remove()
    if selected_value == "Umbralizacion":
        labelTol.grid(column = 0, row = 4, padx = 10, pady = 10)
        entryTol.grid(column = 1, row = 4, padx = 10, pady = 10)
        labelTau.grid(column = 2, row = 4, padx = 10, pady = 10)
        entryTau.grid(column = 3, row = 4, padx = 10, pady = 10)
    elif selected_value == "K-means":
        labelK.grid(column = 0, row = 4, padx = 10, pady = 10)
        entryK.grid(column = 1, row = 4, padx = 10, pady = 10)
    elif selected_value == "Region Growing":
        labelTol.grid(column = 0, row = 4, padx = 10, pady = 10)
        entryTol.grid(column = 1, row = 4, padx = 10, pady = 10)



def open_image_umbralization(image,tol,tau):
    global scale_num
    segmentation = umbralization_segmentation(image, tol, tau) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw()  
    

def open_image_Kmeans(image,k):
    global scale_num

    segmentation = kmeans_segmentation(image, k) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 
    

#El codigo de regio Growing se demora mucho, toca mejorarlo
def open_image_regionGrowing(image,tol):
    global scale_num
    segmentation = region_growing_segmentation(image, tol)
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

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
#---------------------- FRAME DE LAS OPCIONES
optionFrame = Frame(root)
optionFrame.grid(column=0, row=0, sticky="nsew")
optionFrame.config(width=300,height=600, bg='dark turquoise')
#------------------------ FRAME DE LA VISUALIZACION DE LA IMAGEN
imageFrame = Frame(root)
imageFrame.grid(column=1, row=0, sticky="nsew")
imageFrame.config(width=500,height=root["height"], bg='turquoise')

#canvas de la imagen segmentada
fig = plt.figure(figsize=(6, 6), dpi=100)
fig.tight_layout(pad=0)
canvas = FigureCanvasTkAgg(fig, master=imageFrame)
canvas.get_tk_widget().pack()

#canvas de la preview de la imagen
figPre = plt.figure(figsize=(6, 6), dpi=100)
figPre.tight_layout(pad=0)
canvaPre = FigureCanvasTkAgg(figPre, master=optionFrame) 
canvaPre.get_tk_widget().grid(column = 0,row = 7,padx=10,pady=10, columnspan=2) 
canvaPre.get_tk_widget().config(width=200,height=200)
figPre.set_facecolor("#00CED1")




for i in range(10):
    optionFrame.grid_rowconfigure(i, weight=1)
    optionFrame.grid_columnconfigure(i, weight=1)

buttonImageSelector = Button(optionFrame, text="Upload an image .nii.gz",command=recover_image)
buttonImageSelector.grid(column = 0, row = 0,padx=20,pady=10)

buttonImageSelector_target = Button(optionFrame, text="Upload an target image (histogram matching) .nii.gz",command=recover_image_target)
buttonImageSelector_target.grid(column = 1, row = 0, padx = 10 , pady = 10)


labelStandarization = Label(optionFrame, text = "Standarization technique: ",  bg=optionFrame["bg"])
labelStandarization.grid(column = 0, row = 1, padx = 10, pady = 10)
standarizationSelector = ttk.Combobox(optionFrame,values=["Intensity rescaling","Z-score transformation","Histogram matching","White straping"])
standarizationSelector.grid(column = 1, row = 1, padx = 10, pady = 10)
standarizationSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_standarization(selected_image))

labelNoiseRemotion = Label(optionFrame, text = "Noise remotion technique: ",  bg=optionFrame["bg"])
labelNoiseRemotion.grid(column = 0, row = 2, padx = 10, pady = 10)
noiseRemotionSelector = ttk.Combobox(optionFrame,values=["Mean filter","Medium filter"])
noiseRemotionSelector.grid(column = 1, row = 2, padx = 10, pady = 10)
noiseRemotionSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_noise_remotion(image_data))



labelSegmentation = Label(optionFrame, text = "Segmentation algorithm: ",  bg=optionFrame["bg"])
labelSegmentation.grid(column = 0, row = 3, padx = 10, pady = 10)
segmentationSelector = ttk.Combobox(optionFrame,values=["Umbralizacion","K-means","Region Growing"])
segmentationSelector.grid(column = 1, row = 3, padx = 10, pady = 10)
segmentationSelector.bind("<<ComboboxSelected>>", lambda event: segmentation_params())

labelTol = Label(optionFrame, text = "Tolerance:",  bg=optionFrame["bg"])
labelTau = Label(optionFrame, text = "Tau:",  bg=optionFrame["bg"])
labelK = Label(optionFrame, text = "K:",  bg=optionFrame["bg"])

entryTol = Entry(optionFrame)
entryTau = Entry(optionFrame)
entryK = Entry(optionFrame)



scaleWidget = Scale(optionFrame,from_=0,to=48,orient= HORIZONTAL, command=scale_widget_option)
scaleWidget.grid(column=1,row=5,padx=10,pady=20)
labelScale= Label(optionFrame, text = "Z: ",  bg=optionFrame["bg"])
labelScale.grid(column=0,row=5,padx=10,pady=20)

buttonSegmentate = Button(optionFrame, text="Segmentate",command=combo_option_segmentation)
buttonSegmentate.grid(column = 0, row = 6,padx=10,pady=20,columnspan=2)


root.mainloop()