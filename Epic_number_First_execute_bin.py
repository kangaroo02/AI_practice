#!/usr/bin/env python
# coding: utf-8

# In[2]:


# https://www.youtube.com/watch?v=wQ8BIBpya2k
# pip install h5py

import tensorflow as tf
import h5py
import cv2
import numpy as np 
import matplotlib.pyplot as plt

num_model = tf.keras.models.load_model('models/Number_classify-First1570238300.model')


# In[3]:


drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
line_thickness = 16

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=line_thickness)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=line_thickness)        

def img_2_array(img):
    height = img.shape[0]
    width = img.shape[1]

    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    array = np.zeros([height, width])
#     print(type(array))
    
    # array[:] = img[:]
    for i in range(0, height):
        for j in range(0, width):
            array[i][j] = img[i][j]
            
    return array


# img = np.zeros((256,256,3), np.uint8)
# img[:] = (255,255,255)

# ar = img_2_array(img)
# print(len(ar))


# In[10]:



        
img = np.zeros((256,256,3), np.uint8)
img[:] = (255,255,255)
cv2.namedWindow('Number draw')
cv2.setMouseCallback('Number draw',line_drawing)

while(1):
    cv2.imshow('Number draw',img)
    key_in = cv2.waitKey(1) & 0xFF
    if key_in == 27:    # esc
        break
    elif key_in == 13:    #enter
        
        plt.cla()
    # run the desired code
        # change size to 24x24
        img_resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
        img_resize = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
        

        # inver its value black-->0 ; white--> 255
        ret, img_resize = cv2.threshold(img_resize,20,255,cv2.THRESH_BINARY_INV)
        

        img_array = img_2_array(img_resize)
#         img_array = tf.keras.utils.normalize(img_array, axis=2)
        img_array[:,:] = img_array[:,:]/255

        
#         print(img_array)
        
        # plt.imshow(img_array)
        # plt.show()
#         print(img_array.shape)

        predictions = num_model.predict([[img_array]]) # predictions always a list
        print(np.argmax(predictions[0]))
        img[:] = (255,255,255)
        
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:


predictions = num_model.predict([x_test]) #predictions always a list

