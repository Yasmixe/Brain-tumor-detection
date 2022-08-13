import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('webAgg')
import matplotlib.pyplot as plt
import tornado
import tensorflow as tf
import sklearn
import seaborn as sns
import os
from PIL.Image import open
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix , classification_report
#read the data
brain_dt = pd.read_csv('Brain Tumor.csv', usecols=[0, 1]) #showing 2 columns
print(brain_dt.head())
#check if there's une valeur null f data ta3na
print(brain_dt.isna().sum())

#check ch7al kayan men tumeurs ou nn
print(brain_dt['Class'].value_counts())

'''0 : 2079
   1 : 1683'''

#afficher le graphe avec seaborn
sns.countplot(brain_dt['Class'])
#plt.show()

#attribuer chaque image a sa class.
path_list = []
base_path = '../Brain_tumors/Brain Tumor'
for entry in os.listdir(base_path):
    path_list.append(os.path.join(base_path, entry)) # on ajoute a chaque fois les images

#creer la colonne pathes et l'ajouter a notre base de donnees brain_dt
pathes_dict={os.path.splitext(os.path.basename(x))[0]: x for x in path_list}
brain_dt['pathes'] = brain_dt['Image'].map(pathes_dict.get)
#print(brain_dt.head())

#afficher quelques photos

'''for i in range(0, 9):
    img = plt.imread(brain_dt['pathes'][i])
    plt.imshow(img)
    plt.subplot(3, 3, i+1)

print(img.shape)'''
#plt.show()

#ajouter une column pixels
brain_dt['pixels']=brain_dt['pathes'].map(lambda x:np.asarray(open(x).resize((224,224))))

#preprocessing the data
image_list = []
for i in range(0, len(brain_dt)):
    #load images
    brain_img = brain_dt['pixels'][i].astype(np.float32)
    img_array = tf.keras.preprocessing.image.img_to_array(brain_img)
    #append to list of all images
    image_list.append(keras.applications.mobilenet_v2.preprocess_input(img_array))

# convert image list to single array
# Our feature
x = np.array(image_list)
y = np.array(brain_dt.Class)

#splitting the data
#training set 80% 20% test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#build the model mobilenet
num_classes = 1
model = Sequential()
model.add(MobileNetV2(input_shape=(224, 224, 3),weights="imagenet"
                             ,include_top=False))
model.add(keras.layers.GlobalAveragePooling2D())
model.add( Dense(num_classes, activation='sigmoid',name='preds'))
model.layers[0].trainable= False
# show model summary

model.summary()

model.compile(
    # set the loss as binary_crossentropy
    loss=keras.losses.binary_crossentropy,
    # set the optimizer as stochastic gradient descent
    optimizer=keras.optimizers.SGD(lr=0.001),
    # set the metric as accuracy
    metrics=['accuracy']
)

'''
# mock-train the model
model.fit(
    x_train[:,:,:,:],
    y_train[:],
    epochs=110,
    verbose=1,
    validation_data=(x_test[:,:,:,:], y_test[:])
)
model.save("model_brain.h5")
print("Saved model to disk")



# evaluate model on holdout set
eval_score = pretrained_cnn.evaluate(x_test,y_test)
# print loss score
print('Eval loss:',eval_score[0])
# print accuracy score
print('Eval accuracy:',eval_score[1] )
'''
pretrained_cnn = keras.models.load_model('model_brain.h5')
y_pred = (model.predict(x_test) > 0.5).astype("int32")

target_classes = ['No Tumor','Tumor']
print(classification_report(y_test , y_pred , output_dict = True, target_names=target_classes))