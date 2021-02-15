import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as rng

## Read the image 
im = cv.imread('./data/Project1.png')

## Convert image into BW if not
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

## Convert the image into binary image if not already
ret, thresh = cv.threshold(imgray, 127, 255, 0)

## Find the contours of the objects present in the image 
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
## used cv.RETR_TREE to find the heirarchy of images -> discard empty spaces within objects 

## Visuallize all objects 
mask = np.zeros((imgray.shape[0], imgray.shape[1]))
img_mask = cv.drawContours(mask, contours, -1, (255),-1)
plt.imshow(img_mask)
plt.show(block=False)

## Store the contours, bounding rectangle, bounding circle
contours_poly = [None]*len(contours) ## Contour
boundRect = [None]*len(contours) ## Bounding rectangle 
centers = [None]*len(contours) ## Center of bounding circle 
radius = [None]*len(contours) ## Radius of bounding circle

## loop over the different contours
for i, c in enumerate(contours):
    ## If a new object present -> area within the outside boundary
    if hierarchy[0,i,3] == -1:
        contours_poly[i] = cv.approxPolyDP(c, 3, True) ## Exact contour
        boundRect[i] = cv.boundingRect(contours_poly[i]) ## Bounding rectangle 
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i]) ## Center and radius of bounding circle

## dummy image to plot the bounding circles 
drawing = np.zeros((imgray.shape[0], imgray.shape[1], 3), dtype=np.uint8)

## Loop over all the contours detected 
for i, contour in enumerate(contours):    
    if hierarchy[0,i,3] == -1: ## If a new object -> create a new mask
        
        ## Plot the bounding circles
        color = (0, 255, 255)
        drawing = cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        ## last argument given as -1 to fill the contour to get the area of the object 
        drawing = cv.drawContours(drawing, [contour], 0, (255,255,255),-1) ## select all the area within the outside boundary
    
    else: ## If an empty space within the object 
        drawing = cv.drawContours(drawing, [contour], 0, (0, 0, 0),-1) ## unselect all the empty area within the outside boundary


## Plot the bounding circles along with objects 
plt.imshow(drawing)
plt.show(block=False)

## Find the object masks, to be used for Jaccard Similarity 
object_masks = [] ## Store the contours of the different objects 

## Loop over all the contours detected 
for i, contour in enumerate(contours):    
    if hierarchy[0,i,3] == -1: ## If a new object -> create a new mask
        img_mask = np.zeros((im.shape[0], im.shape[1]))
        
        ## last argument given as -1 to fill the contour to get the area of the object 
        img_mask = cv.drawContours(img_mask, [contour], 0, (255),-1) ## select all the area within the outside boundary
    
    else: ## If an empty space within the object 
        img_mask = cv.drawContours(img_mask, [contour], 0, (0),-1) ## unselect all the empty area within the outside boundary
    
    if i+1 == len(contours): ## if it was the last object
        object_masks.append(img_mask) ## store the existing mask 
  
    elif hierarchy[0,i+1,3] == -1: ## if at the next index there is a new object
        object_masks.append(img_mask) ## store the existing mask 


## Find the bounding circle masks, to be used for Jaccard Similarity 
circle_masks = [] ## Store the bounding circles of the different objects

for i in range(len(contours)):

    ## If a new object -> select all the area within the outside boundary
    if hierarchy[0,i,3] == -1:

        ## Create a dummy mask to store the bounding circle mask 
        drawing = np.zeros((imgray.shape[0], imgray.shape[1]), dtype=np.uint8)
        color = (255)

        ## update the mask 
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, -1)
        
        circle_masks.append(drawing) ## Store the mask for further computation 


## Function to compute the Jaccard Similarity score for two image mass
def JaccardSimilarity(mask_1, mask_2):

    ## Flattent the image masks
    bool_mask_1 = mask_1.flatten() > 0
    bool_mask_2 = mask_2.flatten() > 0
    
    ## Compute the numerator for JS
    nr_intersection = float(np.logical_and(bool_mask_1, bool_mask_2).sum())

    ## Compute the denominator for JS
    dr_union = float(np.logical_or(bool_mask_1, bool_mask_2).sum())

    JS_score = nr_intersection / dr_union ## FInal JS score
    
    return JS_score


## Print the cneter coordinates, radius and Jaccard Similarity score for each object and circular bounding box
## Loop over all the contours detected
idx = 0
for i, contour in enumerate(contours):    
    if hierarchy[0,i,3] == -1: ## If a new object -> create a new mask
        drawing = np.zeros((imgray.shape[0], imgray.shape[1], 3), dtype=np.uint8)

        ## Plot the bounding circles
        color = (0, 255, 255)
        drawing = cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        ## last argument given as -1 to fill the contour to get the area of the object 
        caption = 'The co-ordinates of the center of the circular bounding: (' + str(int(centers[i][0])) + ',' +str(int(centers[i][1])) + ')'
        caption += ' \n The radius of the circular bounding:' + str(int(radius[i]))
        
        drawing = cv.drawContours(drawing, [contour], 0, (255,255,255),-1) ## select all the area within the outside boundary
    
    else: ## If an empty space within the object 
        drawing = cv.drawContours(drawing, [contour], 0, (0, 0, 0),-1) ## unselect all the empty area within the outside boundary
    
    if i+1 == len(contours): ## if it was the last object
        
        JS = JaccardSimilarity(object_masks[idx], circle_masks[idx]) ## Calculate the Jaccard Similarity score 
        caption += '\n The Jaccard Simialrity score: ' + str(JS)
        
        ## Plot the object with circulare bounding box
        plt.imshow(drawing)
        plt.figtext(.5, .05, caption, horizontalalignment='center', verticalalignment='center')
        plt.show()
        idx += 1
  
    elif hierarchy[0,i+1,3] == -1: ## if at the next index there is a new object

        JS = JaccardSimilarity(object_masks[idx], circle_masks[idx]) ## Calculate the Jaccard Similarity score 
        caption += '\n The Jaccard Simialrity score: ' + str(JS)
        
        ## Plot the object with circulare bounding box
        plt.imshow(drawing)
        plt.figtext(.5, .05, caption, horizontalalignment='center', verticalalignment='center')
        plt.show()
        idx += 1