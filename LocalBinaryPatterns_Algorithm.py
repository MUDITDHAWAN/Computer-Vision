from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img_dir = './straw.png'

## Load image and display
image = Image.open(img_dir)
# image.show()

## Convert image into numpy array 
np_img = np.array(image.getdata()).reshape(image.size[1], image.size[0])
print("Image Shape:", np_img.shape)

## Plot Numpy array of the image
plt.imshow(np_img, cmap='gray')
plt.show()


## Create histogram for grayscale image
def create_hist(arr, nb_bins=256):
    hist, bin_edges = np.histogram(arr, bins=nb_bins) ## numpy in-built algo to create histogram
    
    ## calculate the center of the histoagram buckets/bins
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
    return hist, bin_centres


## Function that returns a list of neighbour co-ordinates
def find_neighbours(coord):
    neighbours = []
    
    ## Clock-wise direction 
    neighbours.append((coord[0]-1, coord[1]-1)) ## upwards+left pixel
    neighbours.append((coord[0], coord[1]-1)) ## upwards pixel
    neighbours.append((coord[0]+1, coord[1]-1)) ## upwards+right pixel
    neighbours.append((coord[0]+1, coord[1])) ## right pixel
    neighbours.append((coord[0]+1, coord[1]+1)) ## downwards+right pixel
    neighbours.append((coord[0], coord[1]+1)) ## downwards pixel
    neighbours.append((coord[0]-1, coord[1]+1)) ## downwards+left pixel
    neighbours.append((coord[0]-1, coord[1])) ## left pixel
    
    return neighbours 

## Function that computes the binary pattern for the concernes pixel
def create_bit_code(image, coord):
    bit_code = ""
    
    ## loop over the co-ordinates of the concerned pixel
    for neighbour in find_neighbours(coord):
        ## Check with the center pixel 
        if image[coord[0], coord[1]] > image[neighbour]: 
            bit_code += '1'
        else:
            bit_code += '0'
    
    ## return the decimal value for the 8 bit code
    return int(bit_code, base=2)


## Function that computes Local Binary Pattern for each patch 
def patchwise_LBP(image):
    
    ## Creates an empty array to store pattern 
    patch_vals = np.zeros((image.shape[0]-1, image.shape[1]-1)) 
    
    ## Loop over all the pixels present in the current patch
    for row in range(1, image.shape[0]-1): 
        for col in range(1, image.shape[1]-1): 

            bit_code = create_bit_code(image, (row, col)) ## Calculate the decimal value for the concerned pixel in the patch
            patch_vals[row-1, col-1] = bit_code
    
    flatten_vals = patch_vals.ravel() ## Flatten the feature matrix
    
    ## Create the histogram for the flattened LBP feature for the patch
    patch_features, bin_centers = create_hist(flatten_vals, nb_bins=256) 
    
    ### Plot the histogram
    plt.plot(bin_centers, patch_features)
    plt.show()
    
    return patch_vals, patch_features



## Function to create patch wise Local Binary Patterns feature vector of size (nb_patches*256, )
def create_patch(image, nb_patches, block_width, block_height):
    
    feature = [] ## Store the patch wise features for a given block dimension 
    
    nb_patch = 1
    print("Dividing the image into ", nb_patches, "patches.")
    for row in range(0, image.shape[0], block_width): ## strided steps over the width of the image to create patches
        
        for col in range(0, image.shape[1], block_height):## strided steps over the height of the image to create patches
            
            block = image[row:row+block_width, col:col+block_height] ## select a patch of image 
            
            print("For patch:", nb_patch)
            
            ## Calculate the Local Binary Pattern for a patch
            patch_vals, patch_features = patchwise_LBP(block) 
            
            nb_patch += 1
            
            feature.append(patch_features) ## Store the Local Binary Pattern fixed length feature for the patch 
            
    
    print("Number of patches:",len(feature))
    
    print("Number of feature values for each patch:",feature[0].shape)
    
    ## Return the Local Binary Pattern feature of fixed number of patches of the image data 
    return np.concatenate(feature, axis=0)


## Function for Local Binary Patterns feature vector
def Local_Binary_Patterns(image, nb_patches):
        
    ## Calculate the dimension for each patch for a given a number of divisions
    block_width = int(image.shape[0] // (nb_patches**0.5))
    block_height = int(image.shape[1] // (nb_patches**0.5))
        
    ## Calculate the fixed length feature for the given number of spatial divisions
    lbp_feature = create_patch(image, nb_patches, block_width, block_height) 
    
    return lbp_feature

nb_patches = 4
Local_Binary_Pattern_feature = Local_Binary_Patterns(np_img, nb_patches)

print("Local Binary Pattern feature length:", Local_Binary_Pattern_feature.shape) ## (nb_patches*256, )

