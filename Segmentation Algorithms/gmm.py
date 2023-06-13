import numpy as np
#Computes the probability density function of a Gaussian distribution.
def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def gmm(image_data, k, threshold):
    # Initialize parameters
    num_voxels = np.prod(image_data.shape)
    mu = np.linspace(image_data.min(), image_data.max(), k)
    sigma = np.ones(k) * (image_data.max() - image_data.min()) / (2 * k)
    p = np.ones(k) / k
    q = np.zeros((num_voxels, k))
    iteration = 100

    # Run the algorithm
    for i in range(iteration):
        # Calculate responsibilities
        for k in range(k):
            q[:, k] = p[k] * gaussian(image_data.flatten(), mu[k], sigma[k])
        q = q / np.sum(q, axis=1)[:, np.newaxis]

        # Update parameters
        n = np.sum(q, axis=0)
        p = n / num_voxels
        mu = np.sum(q * image_data.flatten()[:, np.newaxis], axis=0) / n
        sigma = np.sqrt(np.sum(q * (image_data.flatten()[:, np.newaxis] - mu) ** 2, axis=0) / n)

        # Check for convergence
        if np.max(np.abs(p - q.sum(axis=0) / num_voxels)) < threshold:
            break

    # Generate segmentation
    segmentation = np.argmax(q, axis=1)
    segmentation = segmentation.reshape(image_data.shape)

    return segmentation