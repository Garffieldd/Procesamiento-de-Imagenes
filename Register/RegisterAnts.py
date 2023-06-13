# from ants import get_ants_data, image_read, resample_image, get_mask, registration, apply_transforms, from_numpy, image_write
# import matplotlib.pyplot as plt
# import numpy as np
# import nibabel as nib
# from tkinter import filedialog

# def register_and_get_image_data(moving):
#     route_fixed_image = filedialog.askopenfilename(filetypes=[("Image files", "FLAIR.nii.gz")])
#     fixed_image = image_read(route_fixed_image)
#     moving_image = image_read(moving)

#     # Perform rigid registration
#     transform = registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

#     # Apply the transformation to the moving image
#     registered_image = apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform['fwdtransforms'])

#     # Convert the registered image to a NumPy array
#     #registered_array = registered_image.numpy()

#     # Plot the registered image using plt.imshow()
#     # # Save the registered image
#     image_write(registered_image, './RegisterResults/Registered_FLAIR.nii.gz')