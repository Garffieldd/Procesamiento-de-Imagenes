def scale_widget_option(value):
    global scale_num
    global canvas
    global canvaPre
    global selected_segmentation
    
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    if canvaPre is not None:
        canvaPre.get_tk_widget().destroy()
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

root=Tk()
root.title("Cargar Imagen")


optionFrame = Frame(root)
optionFrame.grid(column=0, row=0)
optionFrame.config(width=300,height=600, bg='dark turquoise')

scaleWidget = Scale(optionFrame,from_=0,to=48,orient= HORIZONTAL, command=scale_widget_option)
scaleWidget.grid(column=0,row=3,padx=10,pady=100)

root.mainloop()