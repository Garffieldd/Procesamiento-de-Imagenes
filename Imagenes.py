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
from umbralization import umbralization_segmentation
from kmeans import kmeans_segmentation
from region_growing import regios_growing_segmentation

canvas = None
image_nibabel = None
canvaPre = None
scale_num = 0
selected_value = None
preview = None

def recover_image():
    global image_nibabel
    global canvaPre
    global scale_num
    global preview
    if isinstance(canvas, FigureCanvasTkAgg):
        canvas.get_tk_widget().destroy()
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    image_nibabel = nib.load(route)
    preview = image_nibabel.get_fdata()
    create_preview(preview)
    
def create_preview(data):
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(data[:,:,scaleNum])
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvaPre = FigureCanvasTkAgg(fig, master=optionFrame) 
    #canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor='center')
    canvaPre.get_tk_widget().grid(column = 0,row = 4,padx=10,pady=10) 
    canvaPre.get_tk_widget().config(width=200,height=200)
    canvaPre.draw()
    

def combo_option():
    global image_nibabel
    global selected_value
    selected_value = segmentationSelector.get()
    if selected_value is None:
        print("No hay data")
    elif(selected_value == "Umbralizacion"):
        open_image_umbralization(image_nibabel)
    elif(selected_value == "K-means"):
        open_image_Kmeans(image_nibabel)
    elif(selected_value == "Region Growing"):
        open_image_Kmeans(image_nibabel)

def scale_widget_option(value):
    global scale_num
    global canvas
    global canvaPre
    global selected_value
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    if canvaPre is not None:
        canvaPre.get_tk_widget().destroy()
    if image_nibabel is not None:
        scale_num = value

        if selected_value == "Umbralizacion":
            open_image_umbralization(image_nibabel)
            create_preview(preview)
        elif selected_value == "K-means":
            open_image_Kmeans(image_nibabel)
            create_preview(preview)
        elif selected_value == "Region Growing":
            open_image_regionGrowing(image_nibabel)
            create_preview(preview)

def open_image_umbralization(image):
    global canvas
    global scale_num
    if isinstance(canvas, FigureCanvasTkAgg):
        canvas.get_tk_widget().destroy()
    segmentation = umbralization_segmentation(image)   
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas = FigureCanvasTkAgg(fig, master=imageFrame)
    canvas.get_tk_widget().pack()
    canvas.draw()

#Por ahora solo lo hace en 3 separaciones, toca hacerlo dinamico
def open_image_Kmeans(image):
    global canvas
    global scale_num
    if isinstance(canvas, FigureCanvasTkAgg):
        canvas.get_tk_widget().destroy()
    segmentation = kmeans_segmentation(image) 
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas = FigureCanvasTkAgg(fig, master=imageFrame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    

#El codigo de regio Growing se demora mucho, toca mejorarlo
def open_image_regionGrowing(image):
    global canvas
    global scale_num
    segmentation = regios_growing_segmentation(image)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas = FigureCanvasTkAgg(fig, master=imageFrame)
    canvas.get_tk_widget().pack()
     

   
#Creacion de componentes de la interfaz

root=Tk()
root.title("Cargar Imagen")
#root.geometry("800x600")

#---------------------- FRAME DE LAS OPCIONES
optionFrame = Frame(root)
optionFrame.grid(column=0, row=0)
optionFrame.config(width=300,height=600, bg='dark turquoise')
#------------------------ FRAME DE LA VISUALIZACION DE LA IMAGEN
imageFrame = Frame(root)
imageFrame.grid(column=1, row=0)
imageFrame.config(width=500,height=600, bg='turquoise')


for i in range(10):
    optionFrame.grid_rowconfigure(i, weight=1)
    optionFrame.grid_columnconfigure(i, weight=1)
# for i in range(3):
#     imageFrame.grid_rowconfigure(i, weight=1)
#     imageFrame.grid_columnconfigure(i, weight=1)

buttonImageSelector = Button(optionFrame, text="Upload an image .nii.gz",command=recover_image)
buttonImageSelector.grid(column = 0, row = 0,padx=100,pady=10)

segmentationSelector = ttk.Combobox(optionFrame,values=["Umbralizacion","K-means","Region Growing"])
segmentationSelector.grid(column = 0, row = 1, padx = 100, pady = 10)

scaleWidget = Scale(optionFrame,from_=0,to=48,orient= HORIZONTAL, command=scale_widget_option)
scaleWidget.grid(column=0,row=2,padx=10,pady=100)

buttonSegmentate = Button(optionFrame, text="Segmentate",command=combo_option)
buttonSegmentate.grid(column = 0, row = 3,padx=10,pady=100)




root.mainloop()