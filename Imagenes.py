from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
sys.path.append('./Segmentation Algorithms')
sys.path.append('./Standarization')
sys.path.append('./Noise Remotion')
sys.path.append('./Edge detection')
sys.path.append('./Register')
from umbralization import umbralization_segmentation
from kmeans import kmeans_segmentation
from region_growing import region_growing_segmentation
from gmm import gmm
from StandarizationAlgorithms import *
from NoiseRemotion import *
from EdgeDetection import *
#from RegisterAnts import register_and_get_image_data
from register import register_and_get_image_data_itk
from finalRegister import remove_brain
from ttkbootstrap import Style
import os
#canvas = None
image_data = None
scale_num = 0
selected_segmentation = None
preview = np.random.rand(1,1,1)
selected_image = None
selected_image_target = None
fig = None
fig1 = None
tol = None
tau = None
k = None
segmentation = None
#VARIABLES DE LA SEGUNDA PESTAÑA
routeM = None
registration_data = None
scale_num_register = 0
folder_selected = None



# Funcion que se llama cuando se carga la imagen
def recover_image():
    global selected_image
    global fig
    global preview
    global image_data
    global routeM
    routeM = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    selected_image = nib.load(routeM)
    image_data = selected_image.get_fdata()
    preview = selected_image.get_fdata()
    create_preview(preview)  
    update_scale_range()  
    
def recover_image_target():
    global selected_image_target
    global fig
    route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    selected_image_target = nib.load(route)

def combo_option_standarization(image):
    global preview
    global image_data
   
    labelPerc.grid_remove()
    labelPerc.grid_remove()
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
        labelPerc.grid(column = 0, row = 2, padx = 10, pady = 10)
        entryPerc.grid(column = 1, row = 2, padx = 10, pady = 10, sticky='w')
        buttonStandarizateHistogram.grid(column = 1, row = 2, padx = 10, pady = 10 , sticky='e')
        # route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
        # selected_image_target = nib.load(route)
        
    elif(selected_standarization == "White straping"):
        image_data = white_stripe(image)
        preview = white_stripe(image)

def do_histogram_matching(image):
    global preview
    global image_data
    percentile = int(entryPerc.get()) if entryPerc.get() else None
    image_data = histogram_matching(selected_image_target,image,percentile)
    preview = histogram_matching(selected_image_target,image,percentile)
    create_preview(preview)

def combo_option_noise_remotion(data):
    global image_data
    global preview
    selected_sound_remotion = noiseRemotionSelector.get()
    if (selected_sound_remotion == "Mean filter"):
        image_data = mean_filter(data)
        preview = mean_filter(data)
        create_preview(preview)
    elif(selected_sound_remotion == "Medium filter"):
        image_data = median_filter(data)
        preview = median_filter(data)
        create_preview(preview)
    elif(selected_sound_remotion == "Mean filter with borders"):
        image_data = mean_filter_with_borders(data)
        preview = mean_filter_with_borders(data)
        create_preview(preview)
    elif(selected_sound_remotion == "Medium filter with borders"):
        image_data = median_filter_with_borders(data)
        preview = median_filter_with_borders(data)
        create_preview(preview)

def create_preview(data):
    global scale_num
    print(scale_num)
    figPre.clf()
    ax = figPre.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(data[:,:,scaleNum],cmap='gray')
    ax.axis('off')
    canvaPre.draw()  




def combo_option_segmentation():
    global image_data
    global selected_segmentation
    global tol
    global tau
    global k 
    global segmentation
    selected_segmentation = segmentationSelector.get()
    tol = int(entryTol.get()) if entryTol.get() else None
    tau = int(entryTau.get()) if entryTau.get() else None
    k = int(entryK.get()) if entryK.get() else None
    threshold = float(entryTh.get()) if entryTh.get() else None
    if selected_segmentation is None:
        print("No hay data")
        messagebox.showinfo("Error","Debes escojer un tipo de segmentacion")
    elif(selected_segmentation == "Umbralizacion"):
        apply_umbralization(image_data,tol,tau)
        open_image_umbralization()
    elif(selected_segmentation == "K-means"):
        apply_kmeans(image_data,k)
        open_image_Kmeans()
    elif(selected_segmentation == "Region Growing"):
        apply_regionGrowing(image_data,tol)
        open_image_regionGrowing()
    elif(selected_segmentation == "Gmm"):
        apply_gmm(image_data,k,threshold)
        open_image_gmm()
        

def scale_widget_option(value):
    global scale_num
    global selected_segmentation
    global fig
    global tol
    global tau
    global k
    global preview
    global image_data
    if image_data is not None or preview is not None:
        scale_num = value
        create_preview(preview) 
        if selected_segmentation == "Umbralizacion":
            open_image_umbralization()
            
        elif selected_segmentation == "K-means":
            open_image_Kmeans()
            
        elif selected_segmentation == "Region Growing":
            open_image_regionGrowing() 
        elif selected_segmentation == "Gmm":
            open_image_gmm() 


    


def segmentation_params():
    selected_value = segmentationSelector.get()
    labelTol.grid_remove()
    labelTau.grid_remove()
    labelK.grid_remove()
    labelTh.grid_remove()
    entryTol.grid_remove()
    entryTau.grid_remove()
    entryK.grid_remove()
    entryTh.grid_remove()
    if selected_value == "Umbralizacion":
        labelTol.grid(column = 0, row = 6, padx = 10, pady = 10)
        entryTol.grid(column = 1, row = 6, padx = 10, pady = 10)
        labelTau.grid(column = 2, row = 6, padx = 10, pady = 10)
        entryTau.grid(column = 3, row = 6, padx = 10, pady = 10)
    elif selected_value == "K-means":
        labelK.grid(column = 0, row = 6, padx = 10, pady = 10)
        entryK.grid(column = 1, row = 6, padx = 10, pady = 10)
    elif selected_value == "Region Growing":
        labelTol.grid(column = 0, row = 6, padx = 10, pady = 10)
        entryTol.grid(column = 1, row = 6, padx = 10, pady = 10)
    elif selected_value == "Gmm":
        labelK.grid(column = 0, row = 6, padx = 10, pady = 10)
        entryK.grid(column = 1, row = 6, padx = 10, pady = 10)
        labelTh.grid(column = 2, row = 6, padx = 10, pady = 10)
        entryTh.grid(column = 3, row = 6, padx = 10, pady = 10)


def combo_option_edge_detection(data):
    global image_data
    global preview
    selected_value = edgeDetectionSelector.get()
    if selected_value == "Gradient":
        image_data = edge_detection_gradient(data)
        preview = edge_detection_gradient(data)
        create_preview(preview)

def apply_umbralization(image,tol,tau):
    global segmentation
    global selected_image
    segmentation = umbralization_segmentation(image, tol, tau)
    affine = selected_image.affine
    nifti_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
    file_name = os.path.basename(routeM)
    if(file_name == 'FLAIR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultFLAIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'T1.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultT1.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'IR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    else:
        messagebox.showinfo("Aviso","Para que la imagen se guarde, debe de llamarse FLAIR,T1 O IR con su respectiva extension nii.gz")  

def open_image_umbralization():
    global scale_num
    #segmentation = umbralization_segmentation(image, tol, tau) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw()  

def apply_kmeans(image,k):
    global segmentation
    global selected_image
    global routeM
    segmentation = kmeans_segmentation(image, k)  
    affine = selected_image.affine
    nifti_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
    file_name = os.path.basename(routeM)
    if(file_name == 'FLAIR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultFLAIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'T1.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultT1.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'IR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    else:
        messagebox.showinfo("Aviso","Para que la imagen se guarde, debe de llamarse FLAIR,T1 O IR con su respectiva extension nii.gz")

      

def open_image_Kmeans():
    global scale_num
    #segmentation = kmeans_segmentation(image, k) 
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 
  
    
def apply_regionGrowing(image,tol):
    global segmentation
    global selected_image
    segmentation = region_growing_segmentation(image, tol)
    affine = selected_image.affine
    nifti_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
    file_name = os.path.basename(routeM)
    if(file_name == 'FLAIR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultFLAIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'T1.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultT1.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'IR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    else:
        messagebox.showinfo("Aviso","Para que la imagen se guarde, debe de llamarse FLAIR,T1 O IR con su respectiva extension nii.gz")  

def open_image_regionGrowing():
    global scale_num
    #segmentation = region_growing_segmentation(image, tol)
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 

def apply_gmm(image,k,th):
    global segmentation
    global selected_image
    segmentation = gmm(image, k,th)
    affine = selected_image.affine
    nifti_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
    file_name = os.path.basename(routeM)
    if(file_name == 'FLAIR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultFLAIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'T1.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultT1.nii.gz'
        nib.save(nifti_img, output_image_path)  
    elif(file_name == 'IR.nii.gz'):
        output_image_path = './SegmentationResults/SegmentationResultIR.nii.gz'
        nib.save(nifti_img, output_image_path)  
    else:
        messagebox.showinfo("Aviso","Para que la imagen se guarde, debe de llamarse FLAIR,T1 O IR con su respectiva extension nii.gz") 
    
def open_image_gmm():
    global scale_num
    #segmentation = region_growing_segmentation(image, tol)
    fig.clf()
    ax = fig.add_subplot(111)
    scaleNum = int(scale_num)
    ax.imshow(segmentation[:,:,scaleNum])
    ax.axis('off')
    canvas.draw() 

def open_image_register():
    global scale_num_register
    global registration_data
    global folder_selected
    fond = os.path.join(folder_selected, 'FLAIR.nii.gz')
    fond_nib = nib.load(fond)
    fond_data = fond_nib.get_fdata()
    load_registration = nib.load('./RegisterResults/temp_image/FLAIR_skull_lesion.nii.gz')
    registration_data = load_registration.get_fdata()
    scaleNum = int(scale_num_register)
    print(scaleNum)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    #scaleNum = int(scale_num)
    ax1.imshow(registration_data[:,:,scaleNum])
    ax1.imshow(fond_data[:,:,scaleNum], alpha=0.5,cmap='gray')
    ax1.axis('off')
    canvas1.draw() 




def patient_folder():
    global folder_selected
    #selected_value = segmentationSelector.get()
    folder_selected = filedialog.askdirectory(initialdir="/", title="Selecciona una carpeta que contenga sus imagenes IR,T1 y FLAIR") 
    # file_to_load = os.path.join(folder_selected, 'FLAIR.nii.gz')
    # data = nib.load(file_to_load)
    # image_data = data.get_fdata() 
    # print(image_data)   
    


def do_register_T1():
    #selected_value = segmentationSelector.get()
    global folder_selected
    file_to_load = os.path.join(folder_selected, 'T1.nii.gz')
    fixed_image = os.path.join(folder_selected, 'FLAIR.nii.gz')
    segmentation_route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    register_and_get_image_data_itk(file_to_load,segmentation_route,fixed_image)
    
    

def do_register_IR():
    #selected_value = segmentationSelector.get()
    global folder_selected
    file_to_load = os.path.join(folder_selected, 'IR.nii.gz')
    fixed_image = os.path.join(folder_selected, 'FLAIR.nii.gz')
    segmentation_route = filedialog.askopenfilename(filetypes=[("Image files", "*.nii.gz")])
    register_and_get_image_data_itk(file_to_load,segmentation_route,fixed_image)


def do_final_register():
    global folder_selected
    flair_file = os.path.join(folder_selected, 'FLAIR.nii.gz')
    remove_brain('./RegisterResults/Registered_FLAIR_IR.nii.gz','./RegisterResults/Registered_FLAIR_T1.nii.gz',flair_file)
    open_image_register()
    update_scale_range_registration()
    calculate_volumes()

def calculate_volumes():
    #data = np.array(imageDataProfe)
    listbox = Listbox(optionFrame1, width=30)
    listbox.grid(column=1, row=6, padx=10, pady=20)
    listbox.configure(background=optionFrame1["bg"])
    final_image_route = './RegisterResults/temp_image/FLAIR_skull_lesion.nii.gz'
    final_image_nib = nib.load(final_image_route)
    final_image_data = final_image_nib.get_fdata()
    valores_segmentados = np.unique(final_image_data)
    volumenesMM3 = {}
    volumenes = {}
    for valor in valores_segmentados:
        if valor != 0:
            volumen = np.count_nonzero(final_image_data == valor)
            volumenes[valor] = volumen
            tamaño_voxel = np.abs(final_image_nib.affine.diagonal()[:3])
            volumen_en_mm3 = volumen * np.prod(tamaño_voxel)
            volumenesMM3[valor] = volumen_en_mm3
            listbox.insert("end", f"Volumen {int(valor)} : {volumen_en_mm3}")
            listbox.insert("end","\n")

# labelVolume1= Label(optionFrame1, text = "Volumen 1: ")
# labelVolume1.grid(column=1,row=5,padx=10,pady=20)
# labelVolume1.configure(background=optionFrame1["bg"])
# labelVolume1Result= Label(optionFrame1, text = "1")
# labelVolume1Result.grid(column=2,row=5,padx=10,pady=20)
# labelVolume1Result.configure(background=optionFrame1["bg"])
    print(volumenesMM3)
    print(volumenes)
    print(valores_segmentados)



   
def scale_widget_option_register(value):
    global scale_num_register
    if registration_data is not None:
        scale_num_register = value
        open_image_register()

    

def update_scale_range():
    # Obtener el tamaño del eje Z una vez que la imagen se ha cargado
    global preview
    max_z = preview.shape[2] - 1
    scaleWidget.config(to=max_z)

def update_scale_range_registration():
    # Obtener el tamaño del eje Z una vez que la imagen se ha cargado
    global registration_data
    max_z = registration_data.shape[2] - 1
    scaleWidget1.config(to=max_z)

def close_interface():
    plt.close(fig)
    plt.close(figPre)
    plt.close(fig1)
    #plt.close(fig1)
    canvas.get_tk_widget().destroy()
    canvaPre.get_tk_widget().destroy()
    canvas1.get_tk_widget().destroy()
    optionFrame.destroy()
    imageFrame.destroy()
    imageFrame1.destroy()
    
    # Cerrar la ventana principal
    root.destroy()
        
#Creacion de componentes de la interfaz

root=Tk()
root.title("Segmentación de tejidos cerebrales en resonancia magnética")
root.protocol("WM_DELETE_WINDOW", close_interface)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
style = Style(theme='pulse')#lumen pulse simplex
style.configure('TLabel', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12))
# Crear el Notebook para las pestañas
notebook = ttk.Notebook(root)
notebook.pack(fill=BOTH, expand=True)
#-------------- primera pestaña
page1 = ttk.Frame(notebook)
notebook.add(page1, text="Pre-procesate and segmentate")

#---------------------- FRAME DE LAS OPCIONES
optionFrame = Frame(page1)
optionFrame.grid(column=0, row=0, sticky="nsew")
optionFrame.config(width=300,height=600, bg='dark turquoise')
#------------------------ FRAME DE LA VISUALIZACION DE LA IMAGEN
imageFrame = Frame(page1)
imageFrame.grid(column=1, row=0, sticky="nsew")
imageFrame.config(width=500,height=page1.winfo_height(), bg='MidnightBlue')

#canvas de la imagen segmentada
fig = plt.figure(figsize=(6, 6), dpi=100,facecolor="MidnightBlue")
fig.tight_layout(pad=0)
canvas = FigureCanvasTkAgg(fig, master=imageFrame)
canvas.get_tk_widget().pack()

#canvas de la preview de la imagen
figPre = plt.figure(figsize=(6, 6), dpi=100)
figPre.tight_layout(pad=0)
canvaPre = FigureCanvasTkAgg(figPre, master=optionFrame) 
canvaPre.get_tk_widget().grid(column = 0,row = 9,padx=10,pady=10, columnspan=2) 
canvaPre.get_tk_widget().config(width=200,height=200)
figPre.set_facecolor("#00CED1")




for i in range(10):
    optionFrame.grid_rowconfigure(i, weight=1)
    optionFrame.grid_columnconfigure(i, weight=1)

buttonImageSelector = Button(optionFrame, text="Upload an image .nii.gz",command=recover_image)
buttonImageSelector.grid(column = 0, row = 0,padx=20,pady=10)

buttonImageSelector_target = Button(optionFrame, text="Upload an target image (histogram matching) .nii.gz",command=recover_image_target)
buttonImageSelector_target.grid(column = 1, row = 0, padx = 10 , pady = 10)


labelStandarization = Label(optionFrame, text = "Standarization technique: ",  bg='dark turquoise')
labelStandarization.grid(column = 0, row = 1, padx = 10, pady = 10) 
labelStandarization.configure(background=optionFrame["bg"])
standarizationSelector = ttk.Combobox(optionFrame,values=["Intensity rescaling","Z-score transformation","Histogram matching","White straping"])
standarizationSelector.grid(column = 1, row = 1, padx = 10, pady = 10)
standarizationSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_standarization(selected_image))

labelNoiseRemotion = Label(optionFrame, text = "Noise remotion technique: ",  bg=optionFrame["bg"])
labelNoiseRemotion.grid(column = 0, row = 3, padx = 10, pady = 10)
labelNoiseRemotion.configure(background=optionFrame["bg"])
noiseRemotionSelector = ttk.Combobox(optionFrame,values=["Mean filter","Medium filter","Mean filter with borders","Medium filter with borders"])
noiseRemotionSelector.grid(column = 1, row = 3, padx = 10, pady = 10)
noiseRemotionSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_noise_remotion(image_data))


labelEdgeDetection = Label(optionFrame, text = "Edge detection algorithm: ",  bg=optionFrame["bg"])
labelEdgeDetection.grid(column = 0, row = 4, padx = 10, pady = 10)
labelEdgeDetection.configure(background=optionFrame["bg"])
edgeDetectionSelector = ttk.Combobox(optionFrame,values=["Gradient"])
edgeDetectionSelector.grid(column = 1, row = 4, padx = 10, pady = 10)
edgeDetectionSelector.bind("<<ComboboxSelected>>", lambda event: combo_option_edge_detection(image_data))

labelSegmentation = Label(optionFrame, text = "Segmentation algorithm: ",  bg=optionFrame["bg"])
labelSegmentation.grid(column = 0, row = 5, padx = 10, pady = 10)
labelSegmentation.configure(background=optionFrame["bg"])
segmentationSelector = ttk.Combobox(optionFrame,values=["Umbralizacion","K-means","Region Growing","Gmm"])
segmentationSelector.grid(column = 1, row = 5, padx = 10, pady = 10)
segmentationSelector.bind("<<ComboboxSelected>>", lambda event: segmentation_params())

labelPerc = Label(optionFrame, text = "Percentile:",  bg=optionFrame["bg"])
labelPerc.configure(background=optionFrame["bg"])
entryPerc = Entry(optionFrame)

labelTol = Label(optionFrame, text = "Tolerance:")
labelTol.configure(background=optionFrame["bg"])
labelTau = Label(optionFrame, text = "Tau:")
labelTau.configure(background=optionFrame["bg"])
labelK = Label(optionFrame, text = "K:")
labelK.configure(background=optionFrame["bg"])
labelTh = Label(optionFrame, text = "Threshold:")
labelTh.configure(background=optionFrame["bg"])

entryTol = Entry(optionFrame)
entryTau = Entry(optionFrame)
entryK = Entry(optionFrame)
entryTh = Entry(optionFrame)

scaleWidget = Scale(optionFrame,from_=0,to= 0,orient= HORIZONTAL, command=scale_widget_option)
scaleWidget.grid(column=1,row=7,padx=10,pady=20)
labelScale= Label(optionFrame, text = "Z: ")
labelScale.grid(column=0,row=7,padx=10,pady=20)
labelScale.configure(background=optionFrame["bg"])

buttonSegmentate = Button(optionFrame, text="Segmentate",command=combo_option_segmentation)
buttonSegmentate.grid(column = 0, row = 8,padx=10,pady=20,columnspan=2)

buttonStandarizateHistogram = Button(optionFrame, text="Standarizate",command=lambda: do_histogram_matching(selected_image))
#--------------------------------------------SEGUNDA VENTANA SOBRE REGISTRO ------------------------------------------

new_page = ttk.Frame(notebook)
notebook.add(new_page, text="Register")

# Crear el frame de opciones en la nueva pestaña
optionFrame1 = Frame(new_page)
optionFrame1.pack(side=LEFT, fill=BOTH, expand=True)
optionFrame1.config(width=300, height=600, bg='dark turquoise')

# Crear el frame de visualización de imagen en la nueva pestaña
imageFrame1 = Frame(new_page)
imageFrame1.pack(side=RIGHT, fill=BOTH, expand=True)
imageFrame1.config(width=500, height=new_page.winfo_height(), bg='MidnightBlue')

#canvas de la imagen registrada
fig1 = plt.figure(figsize=(6, 6), dpi=100,facecolor="MidnightBlue")
fig1.tight_layout(pad=0)
canvas1 = FigureCanvasTkAgg(fig1, master=imageFrame1)
canvas1.get_tk_widget().pack(fill=BOTH, expand=True)
canvas1.draw()


for i in range(10):
    optionFrame1.grid_rowconfigure(i, weight=1)
    optionFrame1.grid_columnconfigure(i, weight=1)




buttonSelecPatient = Button(optionFrame1,text="Selecciona la carpeta de tu paciente",command=patient_folder)
buttonSelecPatient.grid(column = 1, row = 1,padx=10,pady=20,columnspan=2,sticky="ew")

buttonRegisterT1 = Button(optionFrame1,text="Selecciona la segmentacion T1",command=do_register_T1)
buttonRegisterT1.grid(column = 1, row = 2,padx=10,pady=20,columnspan=2,sticky="ew")

buttonRegisterIR = Button(optionFrame1,text="Selecciona la segmentacion IR",command=do_register_IR)
buttonRegisterIR.grid(column = 1, row = 3,padx=10,pady=20,columnspan=2,sticky="ew")

showVolumes = Button(optionFrame1,text="Mostrar Registro final",command=do_final_register)
showVolumes.grid(column = 1, row = 4,padx=10,pady=20,columnspan=2,sticky="ew")

# labelVolume1= Label(optionFrame1, text = "Volumen 1: ")
# labelVolume1.grid(column=1,row=5,padx=10,pady=20)
# labelVolume1.configure(background=optionFrame1["bg"])
# labelVolume1Result= Label(optionFrame1, text = "1")
# labelVolume1Result.grid(column=2,row=5,padx=10,pady=20)
# labelVolume1Result.configure(background=optionFrame1["bg"])


# labelVolume2= Label(optionFrame1, text = "Volumen 2: ")
# labelVolume2.grid(column=1,row=6,padx=10,pady=20)
# labelVolume2.configure(background=optionFrame1["bg"])
# labelVolume2Result= Label(optionFrame1, text = "2")
# labelVolume2Result.grid(column=2,row=6,padx=10,pady=20)
# labelVolume2Result.configure(background=optionFrame1["bg"])

# labelVolume3= Label(optionFrame1, text = "Volumen 3: ")
# labelVolume3.grid(column=1,row=7,padx=10,pady=20)
# labelVolume3.configure(background=optionFrame1["bg"])
# labelVolume3Result= Label(optionFrame1, text = "3")
# labelVolume3Result.grid(column=2,row=7,padx=10,pady=20)
# labelVolume3Result.configure(background=optionFrame1["bg"])

# labelVolume4= Label(optionFrame1, text = "Volumen 4: ")
# labelVolume4.grid(column=1,row=8,padx=10,pady=20)
# labelVolume4.configure(background=optionFrame1["bg"])
# labelVolume4Result= Label(optionFrame1, text = "4")
# labelVolume4Result.grid(column=2,row=8,padx=10,pady=20)
# labelVolume4Result.configure(background=optionFrame1["bg"])

scaleWidget1 = Scale(optionFrame1,from_=0,to= 0,orient= HORIZONTAL, command=scale_widget_option_register,length=300)
scaleWidget1.grid(column=1,row=5,padx=10,pady=20)
labelScale1= Label(optionFrame1, text = "Z: ")
labelScale1.grid(column=0,row=5,padx=10,pady=20)
labelScale1.configure(background=optionFrame1["bg"])



root.mainloop()