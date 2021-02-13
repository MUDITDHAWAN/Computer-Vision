from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

img_dir = './dog2.png'

## Load image and display
image = Image.open(img_dir)
# image.show()


## Convert image into numpy array 
np_img = np.array(image.getdata()).reshape(image.size[1], image.size[0])
print("Image Shape:", np_img.shape)


## Plot Numpy array of the image
plt.imshow(np_img, cmap='gray')
plt.show()

## Choose the patches 
fg_patch = np_img[150:200, 200:300]
bg_patch = np_img[300:350, 200:300]


## Plot FG patch 
plt.imshow(fg_patch, cmap='gray')
plt.show()

## Plot BG patch 
plt.imshow(bg_patch, cmap='gray')
plt.show()

## Function that creates the likelihood distribution for a given patch 
def patch_prob_dist(patch):
    
    flat_patch = patch.flatten()
    
    hist, bin_edges = np.histogram(flat_patch, bins=range(257)) ## numpy in-built algo to create histogram
    ## [1, 2, 3] -> [1,2) ; [2,3)
    bin_centers = np.arange(0, 256)
    
    hist = hist/ np.max(hist) ## normalizing histogram 

    ## Plot the liikelihood maps
    plt.plot(bin_centers, hist)
    plt.show()
    
    return hist, bin_centers


def spatial_likelihood(image, patch):
    
    ## Create the likelihood distribution for the given patch 
    patch_hist, patch_bin_centers = patch_prob_dist(patch)
    
    ## Plot the liikelihood maps
    plt.plot(patch_bin_centers, patch_hist)
    plt.show()
    
    ## Create a dummy image to store saliency map
    spatial_map = np.ones_like(image).astype(float)

    patch_bin_centers = patch_bin_centers.tolist()
    
    ## Iterate over each pixel value to substitute the likelihood value 
    for row in tqdm(range(image.shape[0])):
        for col in range(image.shape[1]):
            ## Check for the intensity in the bin centers and then use the likelihood
#             intensity = patch_bin_centers.index(image[row,col])
#             spatial_map[row, col] = patch_hist[intensity]

            ## Directly use intensity value as bincenters and grayscale image are integers [0,255]
            spatial_map[row, col] = patch_hist[image[row,col]]

    ## Show the saliency map for the given patch 
    plt.imshow(spatial_map*255, CMAP='gray')
    plt.show()
    
    return spatial_map

## Calculate the FG likelihood intensity distribution and FG saliency map
fg_map = spatial_likelihood(np_img, fg_patch)

## Calculate the BG likelihood intensity distribution and BG saliency map
bg_map = spatial_likelihood(np_img, bg_patch)

## Calculate the final saliency map as a combination of FG and BG
final_map = (fg_map + (1 - bg_map))/2

## Show the final saliency map
plt.imshow(final_map, CMAP='gray')
plt.show()