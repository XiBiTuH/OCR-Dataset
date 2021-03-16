import numpy as np
from PIL import Image
import string
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
from skimage.feature import hog
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tqdm import tqdm
import os
import pickle
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical



def arrangeData():
    '''
    :return: Arranged list of images and respective classes (X,y)
    '''

    #Load data -> 36000 x 50
    data = np.loadtxt("OCRDataSet.txt",usecols=range(50),dtype=float)

    #Get all lower case letters + 0-9 number
    classes = list(string.ascii_lowercase) + ['0','1','2','3','4','5','6','7','8','9']



    X = []
    y = []
    



    #Arrange data
    count_class = 0
    last = 0

    #Go through the whole matrix and for every 1000 iteration append the respective class and image
    for i in range(1000,len(data) + 1000,1000):

        for j in range(last,i,50):

            X.append(data[j:j + 50,:])
            y.append(classes[count_class])

        count_class += 1
        last = i




    return np.asarray(X),np.asarray(y),


#Check if a pixel is white or black
def is_w_o_b(pixel):
    '''
    :param pixel: pixel value to analyse
    :return: return 0 if pixel is black or 255 if value is white
    '''
    if pixel == 255:
        return True
    else:
        return False


def returnCoProps(coMat):

    # In coMat we have an array in which values P[i, j, d, theta] is the number of times that J occurs at a distance of D and an angle of THETA from I


    #SUMMARY OF THE MATRIX

    #Contrast - Returns a measure of the intensity contrast between a pixel and its neighbor over the whole image.
    oriContrast = greycoprops(coMat,prop="contrast")
    #print(f"Contrast : {oriContrast[0][0]}")

    #Homogenity - Returns a value that measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.
    oriHomogeneity = greycoprops(coMat, prop="homogeneity")
    #print(f"Homogeneity : {oriHomogeneity[0][0]}")

    #Energy - Returns the sum of squared elements in the GLCM.
    oriEnergy = greycoprops(coMat, prop="ASM")
    #print(f"Energy : {oriEnergy[0][0]}")

    #Correlation -  Returns a measure of how correlated a pixel is to its neighbor over the whole image
    oriCorrelation = greycoprops(coMat, prop="correlation")
    #print(f"Correlation : {oriCorrelation[0][0]}")

    return oriContrast[0][0], oriHomogeneity[0][0], oriEnergy[0][0], oriCorrelation[0][0]

#Performs feature extractions on an image
def featureExtraction(image):
    '''
    :param image: image to perform feature extraction
    :return: list of characteristics
    '''

    '''
            Freature extraction list
         
         - Number of white pixels  -> C1
         - Number of black pixels -> C2
         - Mean Value of Pixel Values -> C3
         - Horizontal lenght value of the letter/number -> C4 
         - Vertical length value of the letter/number -> C5
         
         - Contrast value of original image -> C6
         - Homogeneity value of original image -> C7
         - Energy value of original image -> C8
         - Correlation value of original image -> C9 
         
         - Contrast value of original image edges -> C10
         - Homogeneity value of original image edges -> C11
         - Energy value of original image edges -> C12
         - Correlation value of original image edges -> C13 
         
         - Contrast value of original image when applied with gabor filter -> C14
         - Homogeneity value of original image when applied with gabor filter -> C15
         - Energy value of original image when applied with gabor filter -> C16
         - Correlation value of original image when applied with gabor filter -> C17 
    '''

    #Image shape
    width, heigth = image.shape

    #Number of white pixels and black pixels
    nWhite = 0
    nBlack = 0
    sumPixels = 0
    image = np.uint8(image)
    im_bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    for pixel in range(width):
        for column in range(heigth):
            if is_w_o_b(im_bw[pixel][column]):
                nWhite += 1
            else:
                nBlack +=1
            sumPixels += im_bw[pixel][column]

    #Check lengths -> Vertical
    vLenght = 0
    s = False
    for pixel in range(width):
        for column in range(heigth):
            if not is_w_o_b(im_bw[pixel][column]):
                s = True
                vLenght = pixel
                break

        if s:
            break

    s = False
    for pixel in range(width - 1, 0,-1):
       for column in range(heigth):
           if not is_w_o_b(im_bw[pixel][column]):
               s = True
               vLenght = pixel - vLenght
               break
       if s:
          break

    # Check lengths -> Horizontal
    hLenght = 0
    s = False
    for pixel in range(width):
      for column in range(heigth):
          if not is_w_o_b(im_bw[column][pixel]):
              s = True
              hLenght = pixel
              break

      if s:
          break

    s = False
    for pixel in range(width - 1, 0, -1):
      for column in range(heigth):
          if not is_w_o_b(im_bw[column][pixel]):
              s = True
              hLenght = pixel - hLenght
              break
      if s:
          break

    #print("Vertical -> " + str(vLenght))
    #print("Horizontal -> " + str(hLenght))

    #Gray co-occurence matrix relative to the original image
    coMat = greycomatrix(image,[2], [0], 256,symmetric=True, normed=True)

    coMatProperties = returnCoProps(coMat)
    #print(coMatProperties)
    
    #EDGE DECETION WITH CANNY
    edges = cv2.Canny(image,100,200)
    coEdges = greycomatrix(edges,[2], [0], 256,symmetric=True, normed=True)
    coEdgesProperties = returnCoProps(coEdges)
    #print(coEdgesProperties)


    #GABOR FILTERS
    gaborFiltR, gaborFiltImg = gabor(image,frequency=0.5)
    gaborFilt = (gaborFiltR ** 2 + gaborFiltImg ** 2) // 2

    #Gray co-occurence matrix relative to the original image
    coGabor = greycomatrix(gaborFilt, [2], [0], 256, symmetric=True, normed=True)
    CoGaborProperties = returnCoProps(coGabor)
    #print(CoGaborProperties)



    #Maybe implement hog features

    #Rename the characteristics
    C1 = nWhite
    #print(f"C1 = {C1}")

    C2 = nBlack
    #print(f"C2 = {C2}")

    C3 = sumPixels / (width * heigth)
    #print(f"C3 = {C3}")

    C4 = hLenght
    #print(f"C4 = {C4}")

    C5 = vLenght
    #print(f"C5 = {C5}")

    C6, C7, C8, C9 = coMatProperties

    C10, C11, C12, C13 = coEdgesProperties

    C14, C15, C16, C17 = CoGaborProperties



    return C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17


def image_to_features(images):

    X = []

    for image in tqdm(range(len(images))):

        chars = featureExtraction(images[image])
        X.append(list(chars))

    X = np.asarray(X)
    pickle.dump(X, open("./features.p","wb"))
    return X



# define NN model
def NNModel():

    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=17, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(36, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    '''
    img = Image.fromarray(X[20]).convert("L")
    img.show()'''

    #Get X and y
    images,y = arrangeData()


    #Encode labels
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    print(y[0])
    dummy_y = to_categorical(y)

    #If features already sabed then load them
    if os.path.exists("./features.p"): X = pickle.load(open("./features.p","rb"))
    else: X = image_to_features(images)

    #Create features families
    labelFamilies = {
        'SimpleFeatures' : X[:6],
        'OriginalFeatures' : X[6:10],
        'EdgeFeatures' : X[10:14],
        'GaborFeatures' : X[14:]
    }


    #Normalize data
    X = preprocessing.normalize(X, norm="l1")

    #Get training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)


    #KNN model
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import plot_confusion_matrix, f1_score
    from sklearn.metrics import accuracy_score

    greatest = [0,0]
    for i in range(1,30):

        #Create model with k = 3
        knnModel = KNeighborsClassifier(n_neighbors=i)

        knnModel.fit(X_train,y_train)

        #Confusion matrix
        #plot_confusion_matrix(knnModel, X_test, y_test)

        y_pred = knnModel.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > greatest[0]:
            greatest[0] = acc
            greatest[1] = i

    print(f"KNN best accuracy : {greatest[0]}")




    #Decision Tree Model
    from sklearn.tree import DecisionTreeClassifier

    treeClassifier = DecisionTreeClassifier(random_state=0)

    treeClassifier.fit(X_train,y_train)

    y_pred = treeClassifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Decision tree accuracy : {acc}")
    #F1 = 2 * (precision * recall) / (precision + recall)
    print(f"Decision tree F1 Score : {f1}")




    #SVM Classifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    svm.fit(X_train,y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Decision tree accuracy : {acc}")


    #Neural Network

    #Get training and testing splits with one hot encoded ys
    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size =0.2)

    nn = NNModel()
    nn.fit(X_train,y_train,epochs=5000)
