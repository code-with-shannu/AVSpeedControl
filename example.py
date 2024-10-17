import cv2 as cv
import numpy as np

import pickle
import pytesseract

#Loading and bluring Image
frame = cv.imread('F:\BTP\TSRExperiment\Exp1\Test/00001.png')
frameBlur = cv.GaussianBlur(frame , (5 , 5) , 0)

#Converting Images to HSV from BGR
hsvBlur = cv.cvtColor(frameBlur , cv.COLOR_BGR2HSV)

#Calculating Range for red color and creating mask to highlight red color
lowerRed1 = np.array([0 , 70 , 50])
upperRed1 = np.array([15 , 255 , 255])

lowerRed2 = np.array([165 , 70 , 50])
upperRed2 = np.array([180 , 255 , 255])

#based on mask use "and" operator to highlight only red and apply canny edge detection on resulting image 
maskBlur1 = cv.inRange(hsvBlur , lowerRed1 , upperRed1)
maskBlur2 = cv.inRange(hsvBlur , lowerRed2 , upperRed2) 

maskBlur = cv.bitwise_or(maskBlur1 , maskBlur2)

res2 = cv.bitwise_and(frameBlur , frameBlur , mask = maskBlur)

edgeDetBlur = cv.Canny(res2 , 100 , 200)

#_ , threshold = cv.threshold(edgeDetBlur , 245 , 255 , cv.THRESH_BINARY_INV)
threshold = cv.adaptiveThreshold(edgeDetBlur , 255 , cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY , 5 , 0)

contours , _ = cv.findContours(threshold , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)


area = []
for cnt in contours: 
    epsilon = 0.01*cv.arcLength(cnt , True)
    approx = cv.approxPolyDP(cnt , epsilon , True)
    cv.drawContours(frameBlur , [approx] , 0 , (0) , 2)

    ar = cv.contourArea(cnt)
    area.append(ar)
    #roi = frame[y:y+h , x:x+w]
    #cv.imwrite('example.png'.format(roi_number) , roi)
    #cv.rectangle(frame.copy() , (x , y) , (x+w , y+h) , (36 , 255 , 12) , 2)
    #roi_number = roi_number + 1
    if(ar > 100):
        x , y , w , h = cv.boundingRect(cnt)
        print('region : ')
        print(x , y , w , h)
        testImage = cv.resize(frame[y:y+h , x:x+w] , (30,30))
        cv.imshow('test' , cv.resize(testImage , (300,300)))
        testImage = cv.cvtColor(testImage , cv.COLOR_BGR2GRAY)
        testImage = cv.equalizeHist(testImage)
        #testImage = testImage/255
        cv.imshow('checking' , cv.resize(testImage , (300,300)))
        testImage = testImage.flatten()

        svm_model = pickle.load(open('tsrModel.sav' , 'rb'))
        prediction = svm_model.predict(testImage.reshape(1 , -1))
        print('prediction : ')
        print(prediction)

        sampleImage = cv.resize(frame[x:x+w , y:y+h] , (30,30))
        cv.imshow('sample' , cv.resize(sampleImage , (300,300)))

    #x , y = approx[0][0]
    #if(len(approx) == 3):
        #cv.putText(frameBlur , "Triangle" , (x,y) , cv.FONT_HERSHEY_COMPLEX , 1 , 0 , 2)
    #elif(len(approx) == 4):
        #cv.putText(frameBlur , "Rectangle" , (x,y) , cv.FONT_HERSHEY_COMPLEX , 1 , 0 , 2)
    #else:
        #cv.putText(frameBlur , "Circle" , (x,y) , cv.FONT_HERSHEY_COMPLEX , 1 , 0 , 2)

print(len(contours) , len(approx))
print(area)

cv.imshow('frame' , frame)
cv.imshow('Blur' , frameBlur)
cv.imshow('mask' , maskBlur)
cv.imshow('res2' , res2)
cv.imshow('edgeBlur' , edgeDetBlur)

k = cv.waitKey() 

cv.destroyAllWindows()