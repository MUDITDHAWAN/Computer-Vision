from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

img_dir = './Project1.png'

## Load image and display
image = Image.open(img_dir)
# image.show()

## Convert image into numpy array 
np_img = np.array(image.getdata()).reshape(image.size[1], image.size[0])
print("Image Shape:", np_img.shape)

## Plot Numpy array of the image
plt.imshow(np_img, cmap='gray')
plt.show()

## Function to find neighbours of a given pixel 
def create_neighbours(coord, max_coord, connectivity=8):
    neighbours = []
    
    right_col = coord[0]+1 < max_coord[0]
    if right_col:
        neighbours.append((coord[0]+1, coord[1])) ## right pixel
    
    left_col = coord[0]-1 >= 0
    if left_col:
        neighbours.append((coord[0]-1, coord[1])) ## left pixel
    
    down_row = coord[1]+1 < max_coord[1]
    if down_row:
        neighbours.append((coord[0], coord[1]+1)) ## downwards pixel
    
    up_row = coord[1]-1 >= 0
    if up_row:
        neighbours.append((coord[0], coord[1]-1)) ## upwards pixel
    
    if connectivity == 8:
        if right_col and up_row:
            neighbours.append((coord[0]+1, coord[1]-1)) ## upwards+right pixel
        
        if right_col and down_row:
            neighbours.append((coord[0]+1, coord[1]+1)) ## downwards+right pixel
        
        if left_col and up_row:
            neighbours.append((coord[0]-1, coord[1]-1)) ## upwards+left pixel
        
        if left_col and down_row:
            neighbours.append((coord[0]-1, coord[1]+1)) ## downwards+left pixel
    
    return neighbours

## Function to check if the neighbour pixel has the same intensity 
def check_neighbours(image, img_mask, start_idx, curr_nb, connectivity=8, val=255):
    # Initializing a queue
    q = Queue()
    q.put(start_idx)

    img_mask[start_idx] = curr_nb ## assign the pixel the current value

    max_coord = image.shape

    while not(q.empty()):
        curr_coord = q.get()
        neighbours = create_neighbours(curr_coord, max_coord, connectivity)
        
        ## Check only the neighbouring pixels if they haven't been assigned to an object before and have the same intensity value 
        connected_neighbours = [neighbour for neighbour in neighbours if (image[neighbour]==val and img_mask[neighbour]==-1)]
        
        ## Add all such pixels in the queue to be explored later (BFS) and change the mask 
        for neighbour in connected_neighbours:
            q.put(neighbour)
            img_mask[neighbour] = curr_nb

## Function to calculate the number of components in a binary image 
def nb_objects_binary_image(image, val=255):

    ## Creates a mask for the image (for visualization)
    mask_img = np.ones_like(image) * -1

    nb_components = 0 ## Store the number of objects in the image 

    ## Loop over all the pixels to check for the presence of an object
    for row in range(image.shape[0]):

        for column in range(image.shape[1]):

            ## If a new white pixel is encountered (not assigned to an object previously)
            if image[row, column] == 255 and mask_img[row, column] == -1:
                nb_components = nb_components + 1 ## Update the object count 

                ## start queue to fill neighbours 
                check_neighbours(image, mask_img, (row, column), nb_components, val=255)
    
    ## The number of objects = total components 
    return nb_components, mask_img        

nb_objects, mask = nb_objects_binary_image(np_img)

print("The number of objects present in the Binary image are", nb_objects)

## Visualize the image mask
plt.imshow(np_img, cmap='gray')
plt.imshow(mask, cmap='jet')
plt.show()