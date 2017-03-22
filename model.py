import csv
import cv2
import numpy as np
import random

lines = []
with open('./training_data/set4/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
numlines = len(lines)
maximages = 60000
val_split = 0.3

correction = 0.25 #Tune this parameter as needed

for line in lines:

	#Randomly skip some images due to memory limits
	if random.random() >= (((maximages*(1+val_split))/6)/numlines):
		continue

	#Load in center, left and right views and apply correction factor
	for i in range(3):
		#0 = center (no correction), 1 = left (+ correction), 2 = right (- correction)
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './training_data/set4/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		if (i == 1):
			calc_correction = correction
		if (i == 2):
			calc_correction = (correction * -1.0)
		if (i == 0):
			calc_correction = 0
		measurement = float(line[3]) + calc_correction
		measurements.append(measurement)


#Augment images by adding flipped images with inverted measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#Import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D

model = Sequential()

#Normalize and crop 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))

#Implement Nvidia style convolutional network as described in "End to End Learning for Self-Driving Cars (2016)" 
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#Use mean square error for this regression network
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=val_split, shuffle=True, nb_epoch=3, verbose=1)

model.save('model_test.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

