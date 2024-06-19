import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
import pickle
import os
import random

def extract_concepts(root_paths, image_id_concepts_dict = dict()):

    for idx, name in enumerate(root_paths):
      with open(name, "r", encoding= 'utf-8-sig') as f:
        reader = csv.reader(f, delimiter = ',')
        
        #if name == 'trainconcept.csv':
        #  image_path = 'train/train/'
        #else:
        #  image_path = 'train/train/'
        if name == 'trainconcept.csv':
          image_path = 'train/'
        else:
          image_path = 'train/'
        for i, line in enumerate(reader):
          if len(line[1]) < 1:
            image_id_concepts_dict[image_path+line[0]+'.jpg'] = []
          else:
            image_id_concepts_dict[image_path+line[0]+'.jpg'] = list(line[1].split(';'))

    return image_id_concepts_dict


def transform_images(path_to_image):
  #path_to_image = os.path.join(training_images_dir, image)
  img = tf.keras.preprocessing.image.load_img(path = path_to_image, target_size= (224,224))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.keras.applications.densenet.preprocess_input(img)

  return img

# Path and csv name to concepts file for training and validation images
path_to_concepts = ['trainconcept.csv','trainconcept.csv']

#Extract concepts for the validation and training images and save to dict
image_id_concepts_dict = extract_concepts(path_to_concepts)
X = []
Y = []
images_ids = []
for image in image_id_concepts_dict.keys():
  X.append(transform_images(image))
  Y.append(image_id_concepts_dict[image])
  images_ids.append(image.split("/")[-1].split("."))
  
X = np.array(X)
Y = np.array(Y)
  
# Use a multilabelbinarizer to transform the concepts into a binary format for training
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y)
print(len(mlb.classes_))

default_densenet = tf.keras.applications.densenet.DenseNet121(include_top=False, weights= 'imagenet') # Load model (only feature extraction part) with imagenet weights
default_densenet.trainable = False # Freeze all layers of the model, so weights remain the same when training, and only weights from added layers update

x = tf.keras.layers.GlobalAveragePooling2D()(default_densenet.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(len(mlb.classes_), activation='sigmoid', name = 'prediction_layer')(x)


bpo_model = tf.keras.models.Model(inputs = default_densenet.input, outputs= x)


init_lr = 1e-4
epochs = 50
batch_size = 32
valid_batch_size = 32

# Objects to be used by the model
opt = tf.keras.optimizers.Adam(learning_rate =init_lr, decay=init_lr / epochs)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5, restore_best_weights= True, mode = 'max')]
bpo_model.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['acc'])

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = 0.2)
train_generator = data_generator.flow(X, Y, batch_size = 32, subset = 'training', seed = 14)
val_generator = data_generator.flow(X, Y, batch_size = 32, subset = 'validation', seed = 14)
history = bpo_model.fit(train_generator, epochs = epochs, validation_data= val_generator, verbose= 1,
                               callbacks = callbacks)
layer_names = [layer.name for layer in default_densenet.layers]

layer_idx = layer_names.index('conv5_block1_0_bn')
for layer in default_densenet.layers[layer_idx:]:
  layer.trainable = True
# A new learning rate is defined, since a keras guide (https://keras.io/guides/transfer_learning/) suggests to lower it. Search for "It's also critical to use a very low learning"
new_lr = 1e-5

# Objects to be used by the model
opt = tf.keras.optimizers.Adam(learning_rate =new_lr, decay=new_lr / epochs)

# Compile model
bpo_model.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['acc'])

history_fined = bpo_model.fit(train_generator, epochs = epochs, validation_data= val_generator, verbose= 1,
                               callbacks = callbacks)
# Now lets use the validation images to create a submission file and evaluate it.
val_x_predict = []
validation_images_path = 'test'

for image in tqdm(os.listdir(validation_images_path), position= 0):
  path_to_image = os.path.join(validation_images_path, image)
  img = tf.keras.preprocessing.image.load_img(path = path_to_image, target_size = (224,224)) # Load actual image
  img = tf.keras.preprocessing.image.img_to_array(img) # Transform image to array of shape (input_shape)
  img = tf.keras.applications.densenet.preprocess_input(img) # This preprocess_input normalizes the pixel values based on imagenet dataset and rescale to a 0-1 values.
  val_x_predict.append(img)

val_x_predict = np.array(val_x_predict) # A numpy array is needed as input for the model
val_predictions = bpo_model.predict(val_x_predict)

# Use previous threshold with better f1-score
val_predictions[val_predictions>=0.1] = 1
val_predictions[val_predictions<0.1] = 0
val_labels_predicted = mlb.inverse_transform(val_predictions)



# The concept(s) are needed as strings separated by ; if applicable
val_labels_united = []
for prediction in val_labels_predicted:
  str_concepts = ''
  for concept in prediction:
    str_concepts += concept+';'
  val_labels_united.append(str_concepts[0:-1])

# The image id needs to be included in the submission
val_images_ids = []
for image in tqdm(os.listdir(validation_images_path), position= 0):
  val_images_ids.append(image.split('.')[0])

# Pass to df  to use to_csv function
predictions_df = pd.DataFrame({'image_ids': val_images_ids})
predictions_df['predictions'] = pd.Series(val_labels_united)
predictions_df.to_csv('bpo-only-bpo-labels.csv', index= False, sep ='\t', header= False) # Dont include headers, and image_id and concepts need to be separated by tab

bpo_model.save('bpo-classifier.h5')

with open("mlb_bpo_classifier.pkl", 'wb') as f:
    pickle.dump(mlb, f)
"""
dp_predictions = pd.read_csv('/content/dp-only-dp-labels.csv', header=None, delimiter='\t', names=['ImageId', 'dp_tags'])
bpo_predictions = pd.read_csv('/content/bpo-only-bpo-labels.csv', header=None, delimiter='\t', names=['ImageId', 'bpo_tags'])
"""

dp_model = tf.keras.models.Model(inputs = default_densenet.input, outputs= x)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 10, restore_best_weights= True, mode = 'max')]
dp_model.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['acc'])

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = 0.2)
train_generator = data_generator.flow(X, Y, batch_size = 32, subset = 'training', seed = 14)
val_generator = data_generator.flow(X, Y, batch_size = 32, subset = 'validation', seed = 14)
history = dp_model.fit(train_generator, epochs = epochs, validation_data= val_generator, verbose= 1,
                               callbacks = callbacks)
layer_names = [layer.name for layer in default_densenet.layers]

layer_idx = layer_names.index('conv5_block1_0_bn')
for layer in default_densenet.layers[layer_idx:]:
  layer.trainable = True
# A new learning rate is defined, since a keras guide (https://keras.io/guides/transfer_learning/) suggests to lower it. Search for "It's also critical to use a very low learning"
new_lr = 1e-5

# Objects to be used by the model
opt = tf.keras.optimizers.Adam(learning_rate =new_lr, decay=new_lr / epochs)

# Compile model
dp_model.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['acc'])

history_fined = dp_model.fit(train_generator, epochs = epochs, validation_data= val_generator, verbose= 1,
                               callbacks = callbacks)
# Now lets use the validation images to create a submission file and evaluate it.
val_x_predict = []
validation_images_path = 'test'

for image in tqdm(os.listdir(validation_images_path), position= 0):
  path_to_image = os.path.join(validation_images_path, image)
  img = tf.keras.preprocessing.image.load_img(path = path_to_image, target_size = (224,224)) # Load actual image
  img = tf.keras.preprocessing.image.img_to_array(img) # Transform image to array of shape (input_shape)
  img = tf.keras.applications.densenet.preprocess_input(img) # This preprocess_input normalizes the pixel values based on imagenet dataset and rescale to a 0-1 values.
  val_x_predict.append(img)

val_x_predict = np.array(val_x_predict) # A numpy array is needed as input for the model
val_predictions = dp_model.predict(val_x_predict)

# Use previous threshold with better f1-score
val_predictions[val_predictions>=0.4] = 1
val_predictions[val_predictions<0.4] = 0
val_labels_predicted = mlb.inverse_transform(val_predictions)



# The concept(s) are needed as strings separated by ; if applicable
val_labels_united = []
for prediction in val_labels_predicted:
  str_concepts = ''
  for concept in prediction:
    str_concepts += concept+';'
  val_labels_united.append(str_concepts[0:-1])

# The image id needs to be included in the submission
val_images_ids = []
for image in tqdm(os.listdir(validation_images_path), position= 0):
  val_images_ids.append(image.split('.')[0])

# Pass to df  to use to_csv function
predictions_df = pd.DataFrame({'image_ids': val_images_ids})
predictions_df['predictions'] = pd.Series(val_labels_united)
predictions_df.to_csv('dp-only-dp-labels.csv', index= False, sep ='\t', header= False) # Dont include headers, and image_id and concepts need to be separated by tab

dp_model.save('dp-classifier.h5')

with open("mlb_dp_classifier.pkl", 'wb') as f:
    pickle.dump(mlb, f)

dp_predictions = pd.read_csv('dp-only-dp-labels.csv', header=None, delimiter='\t', names=['ImageId', 'dp_tags'])
bpo_predictions = pd.read_csv('bpo-only-bpo-labels.csv', header=None, delimiter='\t', names=['ImageId', 'bpo_tags'])


dp_bpo_merged = pd.merge(dp_predictions,bpo_predictions, on='ImageId')

dp_bpo_merged['dp_bpo_tags'] = dp_bpo_merged[dp_bpo_merged.columns[1:]].apply(lambda row: ';'.join(row.dropna()), axis = 1)

dp_bpo_merged.to_csv('bpo-labels.csv', index= False, sep ='\t', header= False, columns=['ImageId','dp_bpo_tags'])




  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
