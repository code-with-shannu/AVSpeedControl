import os
import cv2 as cv
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier

import pickle

print('Loading all folders .........')
root_folder = 'F:\BTP\TSRExperiment\Exp1\Train'
folders = []

for x in range(43):
    folders.append(os.path.join(root_folder , str(x)))

print('Loading Complete ------------------------')
print('Loading Images .........')
images = []
for folder in folders:
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder , filename))
        images.append(img)

print('Loading Complete ---------------------------')
print('ROI Processing and Gray Scaling the Images for Training ........')
trainImages = []
trainFile = pd.read_excel('Train.xlsx')
trainData = pd.DataFrame(trainFile , columns = ['Width' , 'Height' , 'Roi.X1' , 'Roi.Y1' , 'ClassId']).to_numpy()
for i in range(len(images)):
    #trimg = cv.cvtColor(images[i] , cv.COLOR_BGR2HSV)
    trimg = cv.resize(images[i][trainData[i,3]:(trainData[i,3]+trainData[i,1]),trainData[i,2]:(trainData[i,2]+trainData[i,1])],(30,30))
    trimg = cv.cvtColor(trimg , cv.COLOR_BGR2GRAY)
    trimg = cv.equalizeHist(trimg)
    #trimg = trimg/255
    #trimg = np.asarray(trimg)
    trimg = trimg.flatten()
    trainImages.append(trimg)

trainImages = np.array(trainImages)
print('Processing complete ------------------------')
print('Shuffling the dataset for better Accuracy')
shuffled_x , shuffled_y = shuffle(trainImages , trainData[:,4] , random_state = 0)
print('shuffling Complete showing images for confirmation')
print('Reshaping training data for Model .......')
x = shuffled_x
y = shuffled_y

#x = np.reshape(x , (x.shape[0] , -1))
print('Reshaping Complete')
#for i in range(5):
#    cv.imshow('verify' , cv.resize(x[i] , (300 ,300)))
#    cv.waitKey()

#cv.destroyAllWindows()
print('Training the Model..........')
#knn = KNeighborsClassifier(n_neighbors = 7).fit(x , y)
svm_model = SVC(kernel = 'linear' , C = 1).fit(x , y)
#gnb = GaussianNB().fit(x , y)
#dtree = DecisionTreeClassifier().fit(x , y)
#rf = RandomForestClassifier(n_estimators=1000 , max_depth= 10 , random_state=0).fit(x , y)
#nn = MLPClassifier(solver='lbfgs' , alpha=1e-5 , hidden_layer_sizes=(150 , 10) , random_state=0).fit(x , y)
print('Training Complete-------------------------------')

print('saving the model')
modelFile = 'tsrModel.sav'
pickle.dump(svm_model , open(modelFile , 'wb'))
print('model saved')

##testImage = cv.imread('00092.png')
##testImage = cv.resize(testImage , (30 , 30))
##testImage = np.reshape(testImage , (testImage.shape[0] , -1))
##testImage = testImage.flatten()
##print(knn.predict(testImage.reshape(1 , -1)))
##print('Loading Test Images.............')