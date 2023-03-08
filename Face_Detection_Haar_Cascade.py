
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import cv2


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


msd = cv2.imread('C:/Users/user/Documents/DataScience/SkillSlash/Image samples/dhoni.jpg')
messi = cv2.imread('C:/Users/user/Documents/DataScience/SkillSlash/Image samples/messi.jpg')
solvay = cv2.imread('C:/Users/user/Documents/DataScience/SkillSlash/Image samples/face.jpg')


# In[5]:


plt.imshow(msd,cmap='gray')


# In[6]:


plt.imshow(messi,cmap='gray')


# In[7]:


plt.imshow(solvay,cmap='gray')


# # Face Detection

# In[8]:


face_cascade = cv2.CascadeClassifier('C:/Users/user/Documents/DataScience/SkillSlash/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


# In[9]:


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,w+h),(255,255,255),3)
    return face_img


# In[10]:


result_msd = detect_face(msd)


# In[11]:


plt.figure(figsize=(20,10))
plt.imshow(result_msd,cmap='gray')
plt.show()


# In[12]:


result_messi = detect_face(messi)


# In[13]:


plt.figure(figsize=(20,10))
plt.imshow(result_messi,cmap='gray')
plt.show()


# In[14]:


result_solvay = detect_face(solvay)


# In[15]:


plt.figure(figsize=(20,10))
plt.imshow(result_solvay,cmap='gray')
plt.show()


# In[17]:


def adj_detect_face(img):
    face_img=img.copy()
    face_rects=face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,w+h),(255,255,255),3)
    return face_img
    


# # Eye Cascade xml

# In[19]:


eye_cascade=cv2.CascadeClassifier('C:/Users/user/Documents/DataScience/SkillSlash/opencv-master/data/haarcascades/haarcascade_eye.xml')


# In[35]:


def detect_eyes(img):
    face_img=img.copy()
    eyes=eye_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in eyes:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255,255),1)
        return face_img


# In[36]:


result_msd = detect_eyes(msd)


# In[38]:


plt.figure(figsize=(20,10))
plt.imshow(result_msd,cmap='gray')
plt.show(result_msd)


# # Video Face Detection

# In[39]:


cap=cv2.VideoCapture(0)


# In[40]:


while True:
    ret,frame=cap.read(0)
    frame=detect_face(frame)
    cv2.imshow('My_Face_Detection',frame)
    c=cv2.waitKey(1)
    if c== 27:
        break
cap.release()
cv2.destroyAllWindows()

