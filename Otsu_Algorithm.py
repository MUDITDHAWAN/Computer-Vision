from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math 

img_dir = './dog-1020790_960_720.jpg'

image = Image.open(img_dir)

np_img = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
print(np_img.shape)

## Display image 
plt.imshow(np_img)
plt.show()

## Function to convert coloured to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

gray = rgb2gray(np_img) ## Convert to grayscale

## Plot grayscale image
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()

## Create histogram for grayscale image
def create_hist(arr, nb_bins=256):
    hist, bin_edges = np.histogram(arr, bins=nb_bins) ## numpy in-built algo to create histogram
    
    ## calculate the center of the histoagram buckets/bins
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
    return hist, bin_centres

## Otsu Algorithm 
def otsu_algo(image, nbins=256):
    
    flattent_img = image.ravel() ## Flatten the grayscale image 
    
    hist, bin_centers = create_hist(flattent_img, nbins) ## Create the histogram for the flattened grayscale image
    
    hist = hist/ np.sum(hist) ## normalizing histogram (taking out probability)
    
    var_list = [] ## list to store the within class variance for all the threshold values
    
    for idx in range(1,256):
        
        ## calculate within class variance for below the threshold
        left_arr = hist[:idx] ## count values below the threshold
        left_wt = bin_centers[:idx] 
        
        mu_1 = np.average(left_wt, weights=left_arr) ## calculate the weighted mean of the intensities
        
        left_var = np.average((left_wt-mu_1)**2, weights=left_arr) ## calculate the weighted variance of the intensities
        
        ## calculate within class variance for above the threshold
        right_arr = hist[idx:] ## count values above the threshold
        right_wt = bin_centers[idx:] 
        
        mu_2 = np.average(right_wt, weights=right_arr) ## calculate the weighted mean of the intensities

        right_var = np.average((right_wt-mu_2)**2, weights=right_arr) ## calculate the weighted variance of the intensities     
        
        
        ## Calculate the total within vlass-variance for the threshold 
        weighted_var = np.sum(left_arr)*left_var +  np.sum(right_arr)*right_var
        
        var_list.append(weighted_var) ## Store the within-class variance in a list
        
    ## Index for the lowest within-class variance 
    idx_min = var_list.index(min(var_list))
    
    return bin_centers[idx_min] ## return the bin centers for that index i.e., the optimal threshold 
    

threshold = otsu_algo(gray)

print("Threshold Value from Otsu Algorithm:", threshold)

## Plot the threshold along with the histogram for an image
def plot_hist_threshold(image, threshold, nb_bins=256):
    flattent_img = image.ravel() ## Flatten the grayscale image 
    
    hist, bin_centers = create_hist(flattent_img, nb_bins=nb_bins) ## Create the histogram for the flattened grayscale image

    ## Plot the histogram
    plt.plot(bin_centers, hist)
    plt.axvline(threshold, color='b', ls='--') ## plot the threshold
    plt.show()

## Plot the histogram
plot_hist_threshold(gray, threshold)

## Take the assumption to decide bg and fg
def assumptions_fg_bg(mask, pad=10):
    center_val = 0
    center_sum = 0
    boundary_sum = 0
    
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            
            ## Check for boundary assumption with the 'pad' decides the number of rows/ columns 
            ## we take into account while looking at the boundary  
            if (row < pad) or (row >= mask.shape[0]-pad):
                boundary_sum += mask[row, col]
            elif (col < pad) or (col >= mask.shape[1]-pad):
                boundary_sum += mask[row, col]
            
            ## Check for center assumption with the 'pad' decides the number of rows/ columns 
            ## we take into account while looking at the center, more or less the dimension 
            ## of the square area in the center 
            if (row >= math.ceil(mask.shape[0]/2)-(pad/2)) or (row <= math.ceil(mask.shape[0]/2)+(pad/2)):
                if (col <= math.ceil(mask.shape[1]/2)+(pad/2)) or (col >= math.ceil(mask.shape[1]/2)-(pad/2)):
                    center_sum += mask[row, col]
    
    ## Pixels in the center are more of part1 and in the boundary more of part2
    if center_sum >0 and boundary_sum < 0:
        center_val = 1
    
    ## Pixels in the center are more of part2 and in the bounday more of part2
    elif center_sum <0 and boundary_sum > 0:
        center_val = -1
    
    ## Decide on the basis of center assumption if boundary has same number of pixels for 1 and 2
    elif boundary_sum == 0: 
        if center_sum > 0:
            center_val = 1  
        if center_sum < 0:
            center_val = -1 

    ## Decide on the basis of boundary assumption if center has same number of pixels for 1 and 2
    elif center_sum == 0: 
        ## oposite of the boundary majority
        if boundary_sum > 0:
            center_val = -1  
        if boundary_sum < 0:
            center_val = 1 

    ## Both have same ratio of pixels in the boundary and center individually
    elif center_sum == 0 and boundary_sum == 0: 
        center_val = 1 ## Random 
    
    ## If either part 1 or 2 have majority in both center and boundary then chances are that
    ## the the majority object is very big (might be the bg) and the main foreground object 
    ## is smaller and not at the center and boundary
    else: 
        if center_sum > 0:
            center_val = -1
        else:
            center_val = 1

    return center_val

## Plot the bg as blue and coloured fg
def plot_bg_fg(color_img, image, threshold):
    
    ## Create a dummy mask of same size
    mask_img = np.ones_like(image) * -1
    
    ## divide the image into two parts using the threshold 
    mask_img[np.nonzero(image > threshold)] = 1
    
    ## Decide the foreground part
    fg_val = assumptions_fg_bg(mask_img, pad=2)
    
    ## create a bg mask -> 1 for bg part and 0 for fg part 
    bg_mask = np.ones_like(mask_img)
    bg_mask[np.nonzero(mask_img == fg_val)] = 0
    
    ## create a foreground mask -> 1 for fg part and 0 for bg part
    fg_mask = np.ones_like(mask_img)
    fg_mask[np.nonzero(mask_img != fg_val)] = 0
    
    ## Create a masked image --> Blue bg and coloured fg
    I_final = color_img.copy()  # Duplicate image
    I_final[:, :, 0] = I_final[:, :, 0]*fg_mask # red 
    I_final[:, :, 1] = I_final[:, :, 1]*fg_mask # green
    I_final[:, :, 2] = I_final[:, :, 2]*fg_mask + bg_mask*255    # blue
    
    print("Masked Image")
    plt.imshow(I_final)
    plt.show()

## Plot the masked image
plot_bg_fg(np_img, gray, threshold)