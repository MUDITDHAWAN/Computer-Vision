from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_dir = './Blue_Winged_Warbler_0078_161889.jpg'

image = Image.open(img_dir)

np_img = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)


def crop_center(img,cropx,cropy):
    y , x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


cropped_np_img = crop_center(np_img,240,240)
print("Shape of Cropped image:", cropped_np_img.shape)

## Display image 
plt.imshow(cropped_np_img)
plt.show()

## Function to create part wise feature vector of size (nb_parts*nb_channels*nb_statistical_values, )
def pooling(image, nb_parts, block_width, block_height):
    
    feature = [] ## Store the block wise features for a given block dimension 
    
    print("Dividing the image into ", nb_parts, "parts.")
    for row in range(0, image.shape[0], block_width): ## strided steps over the width of the image to create blocks
        
        for col in range(0, image.shape[1], block_height):## strided steps over the height of the image to create blocks
            
            block = image[row:row+block_width, col:col+block_height, :] ## select a block of image with all the channels
            
            block = block.reshape((-1, block.shape[2])) ## Flatten the part of the image for each channel
            
            block_mu = np.mean(block, axis=0) ## Find the mean value for that block for each channel

            block_std = np.std(block, axis=0) ## Find the std value for that block for each channel

            block_feature = np.concatenate((block_mu, block_std), axis=0) ## create a combined feature using the above calculated statistics

            feature.append(block_feature) ## Store the feature for a block
    
    print("Number of parts:",len(feature))
    
    print("Number of statistical values for each part:",feature[0].shape)
    
    ## Return the feature of fixed number of parts pf spatial division of the image data 
    return np.concatenate(feature, axis=0)


## Function for Spatial Pyramid Pooling
def SpatialPyramidPooling(image, list_parts):
    
    fixed_length_feature = [] ## Store the fixed length feature 
    
    for part in list_parts: ## Loop through the list containing the number of spatial divisions
        
        ## Calculate the dimension for each block for a given a number of spatial divisions
        block_width = int(image.shape[0] // (part**0.5))
        block_height = int(image.shape[1] // (part**0.5))
        
        ## Calculate the fixed length feature for the given number of spatial divisions
        feature_part = pooling(image, part, block_width, block_height) 
        
        fixed_length_feature.append(feature_part) ## Store the fixed length feature 
    
    ## Return the fixed legth feature for a given image 
    return np.concatenate(fixed_length_feature, axis=0)


global_feature = SpatialPyramidPooling(cropped_np_img,[4, 9,16,25])
print("Shape of the global feature descriptor for an image through Spatial Pyramid Pooling (SPP):", global_feature.shape)





