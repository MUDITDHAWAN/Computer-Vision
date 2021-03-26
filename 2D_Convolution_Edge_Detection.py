import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from skimage import io, color

image = color.rgb2gray(io.imread(fname="Blue_Winged_Warbler_0078_161889.jpg"))

plt.imshow(image, cmap='gray')
plt.show()


prewitt_filter_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_filter_horizontal = np.transpose(prewitt_filter_vertical)*-1


def conv2d(image, kernel, pad=0, stride=1):
    
    kernel_x = kernel.shape[0]
    kernel_y = kernel.shape[1]
    
    out_x = int((image.shape[0] + 2*pad - kernel_x) / stride) + 1
    out_y = int((image.shape[1] + 2*pad - kernel_y) / stride) + 1
    
    if pad != 0:  
        pad_image = np.zeros((image.shape[0]+2*pad, image.shape[1]+2*pad))
        pad_image[pad:-pad, pad:-pad] = image
    else:
        pad_image = image
    
    output = np.zeros((out_x, out_y))
    
    pbar = tqdm(total=out_x*out_y)
    
    for row in range(0,pad_image.shape[0],stride):
        for col in range(0,pad_image.shape[1],stride):
            
            if row+kernel_x > pad_image.shape[0]: ## out of width
                break
            
            elif col+kernel_y > pad_image.shape[1]: ## out of height
                break
            
            else:
                output[row//stride, col//stride] = (kernel * pad_image[row:row+kernel_x, col:col+kernel_y]).sum()
                pbar.update(1)
                
    pbar.close()
    
    return output


out_vertical = conv2d(image,prewitt_filter_vertical)
plt.imshow(out_vertical, cmap='gray')
plt.show()

out_horizontal = conv2d(image,prewitt_filter_horizontal)
plt.imshow(out_horizontal, cmap='gray')
plt.show()

## Combining the two maps
out_combined = (out_horizontal**2 + out_vertical**2)**0.5
plt.imshow(out_combined, cmap='gray')
plt.show()