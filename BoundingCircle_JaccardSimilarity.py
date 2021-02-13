import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as rng

## Read the image 
im = cv.imread('./Project1.png')

## Convert image into BW if not
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

## Convert the image into binary image if not already
ret, thresh = cv.threshold(imgray, 127, 255, 0)

## Find the contours of the objects present in the image 
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
## used cv.RETR_TREE to find the heirarchy of images -> discard empty spaces within objects 


## Find masks for the objects detected in the image
object_masks = []
for i, contour in enumerate(contours):
    
    if hierarchy[0,i,3] == -1: ## If a new object -> create a new mask
        img_mask = np.zeros((im.shape[0], im.shape[1]))
        
        ## last argument given as -1 to fill the contour to get the area of the object 
        img_mask = cv.drawContours(img_mask, [contour], 0, (255),-1) ## select all the area within the outside boundary
    
    else: ## If an empyt space within the object 
        img_mask = cv.drawContours(img_mask, [contour], 0, (0),-1) ## unselect all the empty area within the outside boundary
    
    if i+1 == len(contours): ## if it was the last object
        object_masks.append(img_mask) ## store the existing mask 
        plt.imshow(img_mask) ## Visuallize a single object
        plt.show()
        
    elif hierarchy[0,i+1,3] == -1: ## if at the next index there is a new object
        object_masks.append(img_mask) ## store the existing mask 
        plt.imshow(img_mask) ## Visuallize a single object
        plt.show()


## Visuallize all objects 
mask = np.zeros((imgray.shape[0], imgray.shape[1]))
img_mask = cv.drawContours(mask, contours, -1, (255),-1)
plt.imshow(img_mask)
plt.show()

## Find the center coordinates and radius for the enclosing circles 
centers = [None]*len(contours)
radius = [None]*len(contours)

## Loop over the detected contours
for i, c in enumerate(contours):
    if hierarchy[0,i,3] == -1: ## Check for a new object
        
        max_x , max_y = np.amax(c, axis=0).ravel() ## Check for the max x and y co-ordinates for the detected object
        
        min_x , min_y = np.amin(c, axis=0).ravel() ## Check for the min x and y co-ordinates for the detected object
        
        candidates = {} ## Store the candidate center point for the circle (radius) 
        
        ## Loop over the pixels situated in the recatangle bounded by the max and min x,y co-ordinates
        for row in range(min_x, max_x+1):
            for col in range(min_y, max_y+1):

                curr_candidate_center = np.array([[row,col]]) ## Check for the pixels in the rectangle 

                ## check the L2 distance for each contour points from the candidate center co-ordinate
                curr_max_dist = np.linalg.norm(c-curr_candidate_center, axis=2) 
                
                ## Store the the maximum radius for each candidate center co-ordinate 
                candidates[(row,col)] = np.max(curr_max_dist) ## (this would be the radius for the particular co-ordinate)
        
        ## The co-ordinate with the minimum radius (this would be the tightest bounding i.e., minimum area)
        centers[i] = min(candidates, key=candidates.get)
        radius[i] = candidates[centers[i]]


## Plot the Bounding Circle and the object together 
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


## Visuallize the object and circle
plt.imshow(drawing)
plt.show()


## Create masks for the bounding circles
circle_masks = []

## Loop over the contours
for i in range(len(contours)):

    if hierarchy[0,i,3] == -1: ## Check for new objects 

        drawing = np.zeros((imgray.shape[0], imgray.shape[1]), dtype=np.uint8) ## Create a dummy mask 

        color = (255)

        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, -1) ## Create the mask for the circle 
        
        ## Store the mask for the enclosing circle 
        circle_masks.append(drawing)

        ## Plot the filled-circle 
        # plt.imshow(drawing, cmap='gray')
        # plt.show()


## the Jaccard Similarity module that takes two binary masks as inputs and outputs the required score.
def JaccardSimilarity(mask_1, mask_2):

    ## Find a boolean mask for the 2 masks
    bool_mask_1 = mask_1 > 0
    bool_mask_2 = mask_2 > 0
    
    ## Find the intersection - Numerator 
    intersection = np.logical_and(bool_mask_1, bool_mask_2).sum()
    
    ## Find the union - Denominator 
    union = float(np.logical_or(bool_mask_1, bool_mask_2).sum())
    
    ## Jaccard Simmilarity Score 
    JS_score = intersection / union
    
    return JS_score


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
        
        JS = JaccardSimilarity(object_masks[idx], circle_masks[idx])
        caption += '\n The Jaccard Similarity score: ' + str(JS)
        print(caption)
        
        plt.imshow(drawing)
        plt.figtext(.5, .01, caption, horizontalalignment='center', verticalalignment='center')
        plt.show()
        idx += 1
  
    elif hierarchy[0,i+1,3] == -1: ## if at the next index there is a new object

        JS = JaccardSimilarity(object_masks[idx], circle_masks[idx])
        caption += '\n The Jaccard Simialrity score: ' + str(JS)
        print(caption)

        plt.imshow(drawing)
        plt.figtext(.5, .01, caption, horizontalalignment='center', verticalalignment='center')
        plt.show()
        idx += 1