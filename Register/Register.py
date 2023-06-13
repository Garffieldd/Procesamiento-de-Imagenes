import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog

def register_and_get_image_data_itk(moving):
    # Load fixed and moving images
    route = filedialog.askopenfilename(filetypes=[("Image files", "FLAIR.nii.gz")])
    fixed_image = sitk.ReadImage(route)
    moving_image = sitk.ReadImage(moving)
    #moving_image_segmentation = sitk.ReadImage(moving_image_path)

    # Convert image types
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    #moving_image_segmentation = sitk.Cast(moving_image_segmentation, sitk.sitkFloat32)

    # Define the registration components
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric - Mutual Information
    registration_method.SetMetricAsMattesMutualInformation()

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Optimizer - Gradient Descent
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                     estimateLearningRate=registration_method.EachIteration)

    # Initial transform - Identity
    initial_transform = sitk.Transform()
    registration_method.SetInitialTransform(initial_transform)

    # Setup for the registration process
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample the moving image to match the fixed image dimensions and orientation
    
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, fixed_image.GetPixelID())

    # Convert the resampled image to Numpy array
    #resampled_array = sitk.GetArrayFromImage(resampled_image)
    
    # Save the resampled image as NIfTI
    output_image_path = './RegisterResults/Registered_FLAIR.nii.gz'
    sitk.WriteImage(registered_image, output_image_path)
