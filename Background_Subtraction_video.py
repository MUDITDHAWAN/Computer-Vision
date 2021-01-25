import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters
from pathlib import Path
from PIL import Image


## Read the video file
cap = cv2.VideoCapture('denis_walk.avi')

## Dimensions, framecount of the video
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

## Empty numpy array to store the video
vid = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))


## Loop over the openCV object
curr_frame = 0
ret = True
while (curr_frame < frameCount  and ret):
    ret, vid[curr_frame] = cap.read()
    curr_frame += 1

cap.release()


## Function to find the Average Frame - using Mean, Median or Mode
def find_bg_frame(vid, avg_type='mean'):
    
    t, h, w, c = vid.shape ## Find video Dimensions 
    flat_vid = vid.reshape((t, -1)) ## Flatten each frame 
    
    if avg_type == 'mean':
        avg_frame_flat = np.mean(flat_vid, axis=0)
    
    elif avg_type == 'mode':
        avg_frame_flat = stats.mode(flat_vid, axis=0).mode
    else:
        avg_frame_flat = np.median(flat_vid, axis=0)
    
    avg_frame = avg_frame_flat.reshape(h, w, c)
    
    print("Background Frame using:", avg_type)
    plt.imshow(avg_frame/255)
    plt.show()
    
    return avg_frame

## Average Frame- using Mean, Median or Mode
bg_frame = find_bg_frame(vid, avg_type='mean')


## FUnction to subtract the background- using Mean, Median or Mode Frame  
def remove_bg(vid, avg_type='mean'):
    
    filtered_vid = []
    ## Find BG frame 
    bg_frame = find_bg_frame(vid, avg_type) ## 3D
    
    for frame_idx in range(vid.shape[0]):
        
        I = vid[frame_idx,:, :,:].copy()
        
        ## Find Change Map (2D)
        change_map = np.mean(abs(vid[frame_idx] - bg_frame), axis=2) ## 2D (Grayscale -- mean change in all channels)
        change_map = change_map/np.max(change_map) ## Change map 2D -- each intensity value [0,1]
        
        ## Plot the change map
        if frame_idx % 10 == 0:
            print("2D Change Map")
            plt.imshow(change_map)
            plt.show()
        
        ## Apply Otsu algorithm to filter out meanigfull changes 
        threshold = filters.threshold_otsu(change_map)
        
        ## Higher intensity in change -> foreground
        fg_mask = change_map >= threshold
        bg_mask = change_map < threshold
        
        ## Mask the image 
        I[:,:,0] = I[:,:,0]*fg_mask + bg_mask*255
        I[:,:,1] = I[:,:,1]*fg_mask + bg_mask*255
        I[:,:,2] = I[:,:,2]*fg_mask 
        
        filtered_vid.append(np.expand_dims(I, axis=0))
        
        ## Plot the foreground
        if frame_idx % 10 == 0:
            print("After background Subtraction")
            plt.imshow(I)
            plt.show()
        
    return np.concatenate(filtered_vid, axis=0)

## Final Frames after background subtraction
final_vid = remove_bg(vid, avg_type='mode')

## Function to save the frames
def save_frames(video, frame_prefix='frame_', dir_name='./output_frames/'):
    
    ## Create directory 
    Path(dir_name).mkdir(exist_ok=True)
    
    ## Loop over all the frames
    for frame_idx in range(video.shape[0]):
        
        image_name = dir_name + frame_prefix + str(frame_idx) + ".jpeg" ## Frame name 
        
        ## Save the file 
        im = Image.fromarray(video[frame_idx])
        im.save(image_name)

save_frames(final_vid, dir_name='./output_frames/')

