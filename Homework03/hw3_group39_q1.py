#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[3]:


import numpy as np
import matplotlib.pyplot as plt_plot
import cv2
from PIL import Image


# #### Viewing the original Image

# In[5]:


image = np.asarray(Image.open('hw3_image.jpeg'))
plt_plot.title("Original Image")
plt_plot.imshow(image)
plt_plot.show()


# ### Below method "implement_PCA(first_N_principal_components, image)" can be used to implement Principal Component Analysis from scratch.
# 
# This method basically takes the list of principal components and image path and then internally computes eigen values and eigen vectors to calculate accumulative variance with the number of components and returns RGB variances.

# ##### Parameters:
# -> first_N_principal_components: list of number of principal components to which the image is to be reduced.<br>
# -> image: path of the input image.

# ##### Returns:
# -> images_after_PCA: list of all the images after implementing PCA wrt principal components.<br>
# -> blue_channel_variances: accumulative variances for blue channel.<br>
# -> green_channel_variances: accumulative variances for green channel.<br>
# -> red_channel_variances: accumulative variances for red channel.<br>

# In[6]:


def implement_PCA(first_N_principal_components, image):
    
    blue_channel_variances = []
    green_channel_variances = []
    red_channel_variances = []
    
    images_after_PCA = []
    combined_accumulative_variances = []
    
    # Executing PCA for all the principal components in the list
    for principal_components in first_N_principal_components:
        # Splitting the image into three RGB channels using CV2
        input_image = cv2.imread(image)
        blue_channel, green_channel, red_channel = cv2.split(input_image)

        # Combining all the three channels in one np array
        combined_channels_RGB = np.array([blue_channel, green_channel, red_channel])

        accumulative_variances = []
        approximations = []
        for i in range(3):
            approximations.append([])

        i = 0
        # Below loop will iterate for all the three color channels and will calculate all the PCA components like eigen values
        # and eigen vectors for all the RGB channels
        for individual_color_channel in combined_channels_RGB:

            #calulating mean for each color channel
            mean_individial_color_channel = np.mean(individual_color_channel)
            mean_individial_color_channel_bar = individual_color_channel - mean_individial_color_channel
            #calculating the covariance for each color channel
            covariance_individual_channel = np.cov(mean_individial_color_channel_bar, bias=False, fweights=None, rowvar = False, aweights=None, y= None)
            eigenvalues_single_color_channel, eigenvectors_single_color_channel = np.linalg.eigh(covariance_individual_channel, UPLO='L')

            # Appending (sum of eigenvalues w.r.t principal components / sum of all eigenvalues) to the list of accumulative variances after reversing the eigen values
            eigenvalues_single_color_channel = np.flip(eigenvalues_single_color_channel, axis=None)
            eigenvectors_single_color_channel = eigenvectors_single_color_channel[:,np.argsort(eigenvalues_single_color_channel, order=None, axis=- 1)]
            accumulative_variances.append(sum(eigenvalues_single_color_channel[0:principal_components])/sum(eigenvalues_single_color_channel))

            # Gather desired number of components and project into original space
            required_principal_components = eigenvectors_single_color_channel[:,0:principal_components]

            # Getting the dot product of both the vectors(mean bar and required principal components)
            individual_dot_vector_product = np.dot(mean_individial_color_channel_bar, required_principal_components, out=None) 

            # Getting the sum of dot product of (eigen vectors and the Transpose of required principal components) and single color channel mean
            dotproduct_mean_sum = np.dot(individual_dot_vector_product, required_principal_components.T, out=None) + mean_individial_color_channel
            approximations[i].append(dotproduct_mean_sum)
            i = i + 1

        # Combine channels into new, compressed image and print image
        final_compressed_image = (np.dstack((approximations[2][0], approximations[1][0], approximations[0][0]))).astype(np.uint8)
        
        
        #combined_accumulative_variances.append(accumulative_variances)
        images_after_PCA.append(final_compressed_image)
        blue_channel_variances.append(accumulative_variances[0])
        green_channel_variances.append(accumulative_variances[1])
        red_channel_variances.append(accumulative_variances[2])
    
    return images_after_PCA, blue_channel_variances, green_channel_variances, red_channel_variances


# #### Calling the method implement_PCA() to get the RGB variances and list of the images after performing PCA
# 

# In[7]:


first_N_principal_components = [1, 4, 20, 50, 100, 200, 300, 400, 800, 1000]
list_of_images_after_PCA, blue_channel_variances, green_channel_variances, red_channel_variances = implement_PCA(first_N_principal_components, 'hw3_image.jpeg')


# ## Using the above values to plot the images

# In[8]:


image_plot = plt_plot.figure(figsize=(20, 12), frameon=True, edgecolor=None)
k = 0
for single_image in list_of_images_after_PCA:
    m = k + 1
    image_plot.add_subplot(4, 4, m)
    plt_plot.imshow(single_image)
    plt_plot.title(f'{first_N_principal_components[k]} Principal Components')
    k = k + 1


# #### After 200 principal components, the image seems to look like the original image, except very few pixels. We can consider that maximum variance is being apprehended in the first 200 components. The same conculsion can be made from the graph below. The RGB variance lines have almost converged after 200 Principal components.

# ### Plotting Principal Components against Accumulative Variance

# In[9]:


scale = 7
plt_plot.plot(first_N_principal_components[0:scale], blue_channel_variances[0:scale], color='blue')
plt_plot.plot(first_N_principal_components[0:scale], green_channel_variances[0:scale], color='green')
plt_plot.plot(first_N_principal_components[0:scale], red_channel_variances[0:scale], color='red')
plt_plot.title('Plot for Accumulative Variance with number of Principal Components')
plt_plot.xlabel('Principal Components')
plt_plot.ylabel('Accumulative Variance')
plt_plot.legend(('Blue Variance', 'Green Variance', 'Red Variance'))


# Maximum of the Variance is being captured by the first 200 components in all the colors, so we have depicted the graph till first 300 components. If we try to plot the graph for more number of components, the scale of x axis will be too high to notice the differences in the three lines for the colors. 
# 
# From the graph we can also deduce that at first 50 components, 90% of the variance is covered. And this is also evident with the images because the image seems to be visible and we can easily make out the different structures in it i.e. the mountains, the man, the cloud, its shadow etc. 

# ### References
# 1. https://drscotthawley.github.io/blog/2019/12/21/PCA-From-Scratch.html
# 2. https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/
# 3. https://www.section.io/engineering-education/image-compression-using-pca/
