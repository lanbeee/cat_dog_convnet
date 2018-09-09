import os, shutil
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image as immm
import numpy as np
from tensorflow.keras.models import load_model

#data has been Downloaded from https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
#Consists of 25000 images of cats and dogs

original_dataset_dir_cat = "C:\\Users\\Synergy\\Desktop\\ML\\datasets\\catsanddogs\\PetImages\\Cat"
original_dataset_dir_dog = "C:\\Users\\Synergy\\Desktop\\ML\\datasets\\catsanddogs\\PetImages\\Dog"
base_dir = "C:\\Users\\Synergy\\Desktop\\ML\\datasets\\cats_and_dogs_small"

cat_files = os.listdir(original_dataset_dir_cat)
curropt_cats = []

dog_files = os.listdir(original_dataset_dir_dog)
curropt_dogs = []

for filename in dog_files :
	if filename.endswith('.jpg'):
		try:
			img = immm.open(os.path.join(original_dataset_dir_dog,filename))
			img.verify()
		except Exception as e:
			curropt_dogs.append(filename)

for filename in cat_files :
	if filename.endswith('.jpg'):
		try:
			img = immm.open(os.path.join(original_dataset_dir_cat,filename))
			img.verify()
		except Exception as e:
			curropt_cats.append(filename)

good_cats = np.setdiff1d(cat_files, curropt_cats)
good_dogs = np.setdiff1d(dog_files, curropt_dogs)

os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = good_cats[:10000]

for fname in fnames:
	src = os.path.join(original_dataset_dir_cat, fname)
	dst = os.path.join(train_cats_dir, 'cat.' + fname)
	shutil.copyfile(src, dst)

fnames = good_cats[10000 : 11250]

for fname in fnames:
	src = os.path.join(original_dataset_dir_cat, fname)
	dst = os.path.join(validation_cats_dir,'cat.' + fname)
	shutil.copyfile(src, dst)

fnames = good_cats[11250 : len(good_cats)]

for fname in fnames:
	src = os.path.join(original_dataset_dir_cat, fname)
	dst = os.path.join(test_cats_dir, 'cat.' + fname)
	shutil.copyfile(src, dst)

fnames = good_dogs[:10000]

for fname in fnames:
	src = os.path.join(original_dataset_dir_dog, fname)
	dst = os.path.join(train_dogs_dir, 'dog.' + fname)
	shutil.copyfile(src, dst)

fnames = good_dogs[10000 : 11250]

for fname in fnames:
	src = os.path.join(original_dataset_dir_dog, fname)
	dst = os.path.join(validation_dogs_dir, 'dog.' + fname)
	shutil.copyfile(src, dst)

fnames = good_dogs[11250 : len(good_dogs)]

for fname in fnames:
	src = os.path.join(original_dataset_dir_dog, fname)
	dst = os.path.join(test_dogs_dir, 'dog.' + fname)
	shutil.copyfile(src, dst)



train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# for data_batch, labels_batch in train_generator:
# 	print('data batch shape:', data_batch.shape)
# 	print('labels batch shape:', labels_batch.shape)
# 	break

model.fit_generator(train_generator, steps_per_epoch=100, epochs=200, validation_data=validation_generator, validation_steps=50, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')])
model.save('cats_and_dogs_model.h5')
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)


# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()