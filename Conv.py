"""
This is about a simple way to run your own kernel, learn how convolution works
You can implement this functions in every code including Machine learning projects 
"""

import numpy as np
import cv2

#Kernels
Contrast = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])
vAxisContrast = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
hAxisContrast = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])

NoKernel = np.array([[0,0,0],
                     [0,1,0],
                     [0,0,0]])
OKernel = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,0]])

BlurKernel = np.ones((3,3))


#Import an image
"""
Input = cv2.imread("car5.png")
Input = cv2.cvtColor(Input, cv2.COLOR_BGR2GRAY)
Input = np.array(Input)
"""
#Convolution one kernel and dual kernel
"""
frame = Conv(frame, vAxis)

frame = ImageAND(Conv(frame,vAxisContrast),Conv(frame,hAxisContrast))

"""
#Save image
"""
cv2.imwrite("filename.png", frame)

"""


def Conv(Image, kernel, div = 1):
    xSize = Image.shape[0]
    ySize = Image.shape[1]
    image = np.zeros(Image.shape)
    for x in range(xSize-1):
        for y in range(ySize-1):
            
            if x > 0 and x < xSize and y > 0 and y < ySize:
                image[x,y] = ((Image[x-1][y-1] * kernel[0][0]) + (Image[x][y-1] * kernel[1][0]) + (Image[x+1][y-1] * kernel[2][0]) + (Image[x-1][y] * kernel[0][1]) + (Image[x][y] * kernel[1][1]) + (Image[x+1][y] * kernel[2][1]) + (Image[x-1][y+1] * kernel[0][2]) + (Image[x][y+1] * kernel[1][2]) + (Image[x+1][y+1] * kernel[2][2]))/div
            
            

    return image
            
def ImageAND(img1,img2, round = 0):
    xSize = img1.shape[0]
    ySize = img1.shape[1]
    image = np.zeros((xSize,ySize))
    for x in range(xSize):
        for y in range(ySize):
            
            image[x,y] = img1[x,y] if img1[x,y] > img2[x,y] else img2[x,y]
            #image[x,y] = img1[x,y] + img2[x,y]
            if round == 1:

                image[x,y] = 255 if image[x,y] > 50 else 0
            elif round == 2:
                image[x,y] = 0 if image[x,y] > 50 else 255

    return image    


def Camera( cam ):
    cap = cv2.VideoCapture(cam)

    while True:
        ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (300,300))
        frame = ImageAND(Conv(frame,vAxisContrast),Conv(frame,hAxisContrast))
        #frame = Conv(frame, vAxis)
        cv2.imshow("frame",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
