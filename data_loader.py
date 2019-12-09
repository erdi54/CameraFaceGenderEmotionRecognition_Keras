import numpy as np
import pandas as pd
from scipy.io import loadmat
from random import shuffle
import cv2

emotion_dataset_path = './dataset/fer2013.csv'
gender_dataset_path = './dataset/imdb.mat'
image_size = (48, 48)

def emotion_load_dataset():

     data = pd.read_csv(emotion_dataset_path)
     pixels = data['pixels'].tolist()
     width, height = 48, 48
     faces = []
     for pixel_sequence in pixels:
         face = [int(pixel) for pixel in pixel_sequence.split(' ')]
         face = np.asanyarray(face).reshape(width, height)
         face = cv2.resize(face.astype('uint8'),image_size)
         faces.append(face.astype('float32'))
     faces = np.asarray(faces)
     faces = np.expand_dims(faces, -1)
     emotions = pd.get_dummies(data['emotion']).as_matrix()

     return faces, emotions

def preprocess_input(x ,v2=True):

     x = x.astype('float32')
     x = x / 255.0

     if v2:
        x = x - 0.5
        x = x * 2.0

     return x

def gender_load_dataset (validation_split=.2, do_shuffle=False):
    face_score_treshold = 3
    dataset = loadmat(gender_dataset_path)
    image_names_array = dataset['imdb']['full_path'][0, 0][0]
    gender_classes = dataset['imdb']['gender'][0, 0][0]
    face_score = dataset['imdb']['face_score'][0, 0][0]
    second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
    face_score_mask = face_score > face_score_treshold
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)
    image_names_array = image_names_array[mask]
    gender_classes = gender_classes[mask].tolist()
    image_names = []
    for image_name_arg in range(image_names_array.shape[0]):
        image_name = image_names_array[image_name_arg][0]
        image_names.append(image_name)
    ground_truth_data = dict(zip(image_names, gender_classes))
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle is not False:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys, ground_truth_data

