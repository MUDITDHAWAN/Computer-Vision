import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.notebook import tqdm


import skimage.io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import measure
from skimage.color import label2rgb


# read image
image = skimage.io.imread(fname="Black_Footed_Albatross_0009_34.jpg")


## find slic superpixels for the image
segments_slic = slic(image, n_segments=250, sigma=1, start_label=1)


## show the boundaries of the super pixels
plt.imshow(mark_boundaries(image, segments_slic))
plt.show()


## show the superpixels as the average intensity 
plt.imshow(label2rgb(segments_slic, image, kind='avg')/255)
plt.show()

## find proporties for each region/super-pixel
regions = measure.regionprops(segments_slic, intensity_image=image)


## [r.centroid for r in regions] --> centroid for each super pixel
## [r.mean_intensity for r in regions] --> mean colour for each super pixel


superpixel_saliency_dict = {}   ## store the saliency value for each super pixel

## iterate over each super pixel
for idx, curr_region in tqdm(enumerate(regions), total=len(regions)):
    
    superpixel_saliency_dict[idx] = 0
        
    curr_superpixel_coord = curr_region.centroid ## center of the superpixel
    curr_superpixel_intensity = curr_region.mean_intensity ## average colour of the super pixel
    
    ## iterate over each super-pixel to calculate the saliency value for current super-pixel
    for region in regions:
        ## find the colour distance
        colour_dist = np.linalg.norm(curr_superpixel_intensity-region.mean_intensity)
        
        ## find e^ (the spatial distance of the super pixels)
        spatial_dist = math.exp(-(abs(curr_superpixel_coord[0] - region.centroid[0])+ abs(curr_superpixel_coord[1] - region.centroid[1])) / 512) 
        
        ## store the value-- summation over all superpixels
        superpixel_saliency_dict[idx] += (colour_dist*spatial_dist)


saliency_matrix = np.zeros((image.shape[0], image.shape[1])) ## dummy image to display saliency map

for idx, region in enumerate(regions): ## go over all the superpixels 
    
    for coord in region.coords: ## iterate over the pixels included in the superpixel and assign the saliency value 
        saliency_matrix[coord[0], coord[1]] = superpixel_saliency_dict[idx]

## Normalize the saliency values
saliency_matrix = saliency_matrix / np.max(saliency_matrix)


## display the saliency map 
plt.imshow(saliency_matrix, cmap='gray')
plt.show()