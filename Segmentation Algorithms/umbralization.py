import numpy as np
def umbralization_segmentation(image):
    image_data = image.get_fdata()
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
    return segmentation  