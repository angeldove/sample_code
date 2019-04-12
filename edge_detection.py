import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
  
# reading image data
img = cv2.imread('images/004_new.jpeg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# converting BGR to HSV 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
      
# define range of red color in HSV 
lower_red = np.array([30,150,50]) 
upper_red = np.array([255,255,180]) 
      
# create a red HSV colour boundary and  
# threshold HSV image 
mask = cv2.inRange(hsv, lower_red, upper_red) 
  
# Bitwise-AND mask and original image 
res = cv2.bitwise_and(img,img, mask= mask) 
  
# Display an original image 
# cv2.imshow('Original',frame) 
  
# finds edges in the input image image and 
# marks them in the output map edges 
edges = cv2.Canny(img,100,200) 
  
# Display edges in a frame 
# cv2.imshow('Edges',edges) 
# cv2.waitKey(0)

_, threshold = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     print(len(approx))
#     if len(approx)==6:
#         print("hexgonal")
#         cv2.drawContours(img,[cnt],0,255,-1)
#     elif len(approx)==5:
#         print ("pentagon")
#         cv2.drawContours(img,[cnt],0,(0,255,0),-1)
#     elif len(approx)==4:
#         print ("square")
#         cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#     elif len(approx) == 9:
#         print ("half-circle")
#         cv2.drawContours(img,[cnt],0,(255,255,0),-1)
#     elif len(approx) > 15:
#         print ("circle")
#         cv2.drawContours(img,[cnt],0,(0,255,255),-1)

cv2.drawContours(img, contours, -1, (0, 0, 255), -1)
cv2.imshow('img',img)
cv2.waitKey(0)

  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  