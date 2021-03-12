#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.notebook import tqdm
import skimage.io
from sklearn.cluster import KMeans


# In[2]:


# read image
image1 = skimage.io.imread(fname="Cap1.png")
image2 = skimage.io.imread(fname="Cap2.png")


# In[3]:


plt.imshow(image1)
plt.show()


# In[4]:


plt.imshow(image2)
plt.show()


# In[6]:


multiple_images = list(image1[:,:,:3].reshape((image1.shape[0] * image1.shape[1], 3)))


# In[7]:


multiple_images.extend(list(image2[:,:,:3].reshape((image2.shape[0] * image2.shape[1], 3))))


# In[8]:


multiple_images = np.array(multiple_images)


# In[10]:


clf = KMeans(n_clusters = 7)
clf.fit(multiple_images)


# In[14]:


clustered = clf.cluster_centers_[clf.labels_]


# In[15]:


clustered_image1 = clustered[:image1.shape[0] * image1.shape[1], :]
clustered_image2 = clustered[image1.shape[0] * image1.shape[1]:, :]


# In[16]:


# Reshape back the image from 2D to 3D image
clustered_3D = clustered_image1.reshape(image1.shape[0], image1.shape[1], 3)
plt.imshow(clustered_3D/255)
plt.title('Clustered Image')
plt.show()


# In[17]:


# Reshape back the image from 2D to 3D image
clustered_3D = clustered_image2.reshape(image2.shape[0], image2.shape[1], 3)
plt.imshow(clustered_3D/255)
plt.title('Clustered Image')
plt.show()


# In[18]:


clf.labels_.shape


# In[19]:


image1_labels = clf.labels_[:image1.shape[0] * image1.shape[1]]
image2_labels = clf.labels_[image1.shape[0] * image1.shape[1]:]


# In[20]:


# Reshape back the image from 2D to 3D image
clustered_3D = image1_labels.reshape(image1.shape[0], image1.shape[1])
plt.imshow(clustered_3D)
plt.title('Clustered Image')
plt.show()


# In[21]:


unique1, counts1 = np.unique(image1_labels, return_counts=True)
dict1 = dict(zip(unique1, counts1))


# In[22]:


print(dict1)


# In[23]:


unique2, counts2 = np.unique(image2_labels, return_counts=True)
dict2 = dict(zip(unique2, counts2))


# In[24]:


print(dict2)


# In[142]:


corresponding_hist = {}
for clust_center in set(list(dict1.keys()) + list(dict2.keys())):
    corresponding_hist[clust_center] = {}
    for idx, image_dict in enumerate([dict1, dict2]):
        corresponding_hist[clust_center][idx] = image_dict[clust_center]


# In[152]:


corresponding_hist


# In[146]:


corresponding_cue = []
for k in list(corresponding_hist.keys()):
    bin_heights = [corresponding_hist[k][j] for j in corresponding_hist[k].keys()]
    
    corresponding_cue.append(bin_heights)

corresponding_cue = np.array(corresponding_cue)
print(corresponding_cue.shape)
corresponding_cue = np.var(corresponding_cue, axis=1)
print(corresponding_cue.shape)
corresponding_cue = corresponding_cue + 1

corresponding_cue = 1/ corresponding_cue


# In[148]:


corresponding_cue = corresponding_cue/ np.max(corresponding_cue)


# In[150]:


corresponding_feature_map = corresponding_cue[clf.labels_]
image1_corresponding = corresponding_feature_map[:image1.shape[0] * image1.shape[1]].reshape(image1.shape[0], image1.shape[1])
image2_corresponding = corresponding_feature_map[image1.shape[0] * image1.shape[1]:].reshape(image2.shape[0], image2.shape[1])


# In[151]:


plt.imshow(image1_corresponding, cmap='gray')
plt.show()


# In[156]:


plt.imshow(image2_corresponding, cmap='gray')
plt.show()


# In[27]:


clf.cluster_centers_


# In[28]:


_, weights = np.unique(clf.labels_, return_counts=True) 
weights = weights / clf.labels_.shape[0]


# In[29]:


clf.cluster_centers_


# In[30]:


contrast_cue = []
for k in range(clf.cluster_centers_.shape[0]):
    
    curr_cluster_center = clf.cluster_centers_[k,:].reshape((1,3))
    
    print(curr_cluster_center)
    
    repeated = np.repeat(curr_cluster_center, repeats=clf.cluster_centers_.shape[0], axis=0)
    
    print(repeated.shape)
    
    dist_vector = np.linalg.norm(repeated - clf.cluster_centers_, axis=1)
    
    print(dist_vector)
    
    contrast_cue.append(np.average(dist_vector, axis=0, weights=weights))


# In[138]:


contrast_cue


# In[155]:


contrast_feature_map = np.array(contrast_cue)[clf.labels_]
image1_contrast = contrast_feature_map[:image1.shape[0] * image1.shape[1]].reshape(image1.shape[0], image1.shape[1])
image2_correspondingcontrast = contrast_feature_map[image1.shape[0] * image1.shape[1]:].reshape(image2.shape[0], image2.shape[1])


# In[157]:


plt.imshow(image1_contrast, cmap='gray')
plt.show()


# In[158]:


plt.imshow(image2_contrast, cmap='gray')
plt.show()


# In[140]:


from math import pi
def calculate_normal_distance(x, mean=0, var=0.01):
    exponent = np.exp(-((x-mean)**2 / (2 * var )))

    return (1 / (np.sqrt(2 * pi* var))) * exponent


# In[126]:


images = [image1, image2]
total_normal_loc_images = []
for idx in range(len(images)):
    
    image_locs = np.zeros((images[idx].shape[0], images[idx].shape[1], 2))
    
    for row in range(image_locs.shape[0]):
        for col in range(image_locs.shape[1]):
            image_locs[row, col, 0] =  row/images[idx].shape[0] 
            image_locs[row, col, 1] =  col/images[idx].shape[1]
    
    center_image_loc = np.ones((images[idx].shape[0], images[idx].shape[1], 2))
    center_image_loc[:,:,0] = center_image_loc[:,:,0] * (images[idx].shape[0] // 2) / images[idx].shape[0]
    center_image_loc[:,:,1] = center_image_loc[:,:,1] * (images[idx].shape[1] // 2) / images[idx].shape[1]
    
    dist_image_locs = (np.linalg.norm(image_locs - center_image_loc, axis=2))**2
    plt.imshow(dist_image_locs, cmap='gray')
    plt.show()
    
    dist_image_locs = calculate_normal_distance(dist_image_locs)
    plt.imshow(dist_image_locs, cmap='gray')
    plt.show()
    print(dist_image_locs.shape)
    
    total_normal_loc_images.append(dist_image_locs.flatten())

total_normal_loc_images = np.concatenate(total_normal_loc_images, axis=0)


# In[127]:


spatial_cue = []
_, weights = np.unique(clf.labels_, return_counts=True) 

for k in range(clf.cluster_centers_.shape[0]):
    spatial_cue.append(np.sum(total_normal_loc_images[clf.labels_ == k]) / weights[k])


# In[128]:


spatial_cue


# In[129]:


spatial_feature_map = np.array(spatial_cue)[clf.labels_]


# In[130]:


image1_spatial = spatial_feature_map[:image1.shape[0] * image1.shape[1]].reshape(image1.shape[0], image1.shape[1])
image2_spatial = spatial_feature_map[image1.shape[0] * image1.shape[1]:].reshape(image2.shape[0], image2.shape[1])


# In[131]:


plt.imshow(image1_spatial, cmap='gray')
plt.show()


# In[132]:


plt.imshow(image2_spatial, cmap='gray')
plt.show()


# In[133]:


saliency_feature = np.array(spatial_cue) * np.array(contrast_cue)


# In[134]:


saliency_map = saliency_feature[clf.labels_]


# In[135]:


image1_saliency = saliency_map[:image1.shape[0] * image1.shape[1]].reshape(image1.shape[0], image1.shape[1])
image2_saliency = saliency_map[image1.shape[0] * image1.shape[1]:].reshape(image2.shape[0], image2.shape[1])


# In[136]:


plt.imshow(image1_saliency/np.max(image1_saliency), cmap='gray')
plt.show()


# In[137]:


plt.imshow(image2_saliency/np.max(image2_saliency), cmap='gray')
plt.show()

