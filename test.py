import cv2 as cv
import pickle
import pandas as pd
import os
import numpy as np

testFileImages = []
for filename in os.listdir('F:\BTP\TSRExperiment\Exp1\Test'):
    img = cv.imread(os.path.join('F:\BTP\TSRExperiment\Exp1\Test' , filename))
    testFileImages.append(img)
print('Loading complete----------------------------------------------------------')

print('Resizing and GrayScaling the images for testing...................')
testFile = pd.read_excel('Test.xlsx')
testData = pd.DataFrame(testFile , columns = ['Width' , 'Height' , 'Roi.X1' , 'Roi.Y1' , 'ClassId']).to_numpy()

testImages = []
check = []
for i in range(len(testFileImages)):
    teimg = cv.resize(testFileImages[i][testData[i,3]:(testData[i,3]+testData[i,1]),testData[i,2]:(testData[i,2]+testData[i,1])],(30,30))
    teimg = cv.cvtColor(teimg , cv.COLOR_BGR2GRAY)
    teimg = cv.equalizeHist(teimg)
    check.append(teimg)
    teimg = teimg.flatten()
    testImages.append(teimg)

testImages = np.array(testImages)
print('Resizing complete------------')

print('Reshaping the testData for model...............')
x_test = testImages
##x_test = np.reshape(x_test , (x_test.shape[0] , -1))
y_test = testData[:,4]
print('Reshaping Complete---------------------------------')

i = 1

svm_model = pickle.load(open('tsrModel.sav' , 'rb'))
prediction = svm_model.predict(x_test[i].reshape(1 , -1))
print('prediction : ')
print(prediction)

#accuray = svm_model.score(x_test , y_test)
#print(accuray)

cv.imshow('check' , cv.resize(check[i] , (300,300)))

cv.waitKey()
cv.destroyAllWindows()