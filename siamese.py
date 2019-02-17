import os
import datetime
import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from tqdm import tqdm


def dataset_load(path, img_res):
    dir_list = sorted(os.listdir(path))
    y_vector = []
    count = 0
    tot_files = sum([len(files) for r, d, files in os.walk(path)])
    data = np.ndarray(shape=(tot_files, img_res[0], img_res[1], 3), dtype=np.float32)
    for id_class, dir in enumerate(dir_list):
        img_list = os.listdir("{0}/{1}".format(path, dir))
        for id, img_name in enumerate(img_list):
            img = cv2.imread("{0}/{1}/{2}".format(path, dir, img_name))
            img = cv2.resize(img, (img_res[0], img_res[1]))
            img = img_to_array(img)
            img = img / 255
            data[count] = img
            count += 1
            y_vector.append(id_class)
    return data, y_vector


def generate_permutation_dataset(dataset_x, dataset_y, img_res):
    num_examples = len(set(dataset_y))
    array_shape = dataset_x.shape[0] * num_examples * 2
    first_image = np.ndarray(shape=(array_shape, img_res[0], img_res[1], 3), dtype=np.float32)
    second_image = np.ndarray(shape=(array_shape, img_res[0], img_res[1], 3), dtype=np.float32)
    mutual_score = []

    indices_dic_same = {un: [num for num, item in enumerate(dataset_y) if item == un] for un in set(dataset_y)}
    indices_dic_diff = {un: [num for num, item in enumerate(dataset_y) if item != un] for un in set(dataset_y)}

    count = 0

    for num, y in enumerate(tqdm(dataset_y)):
        for _ in range(num_examples):
            first_image[count, ...] = dataset_x[num, ...]
            second_image[count, ...] = dataset_x[np.random.choice(indices_dic_same[y], 1)[0], ...]
            mutual_score.append(1)
            count += 1
            first_image[count, ...] = dataset_x[num, ...]
            second_image[count, ...] = dataset_x[np.random.choice(indices_dic_diff[y], 1)[0], ...]
            mutual_score.append(0)
            count += 1
    return first_image, second_image, mutual_score


def plot_permutation_group(dataset_first, dataset_second, score):
    indices_plot = np.random.choice(range(dataset_first.shape[0]), 8)
    fig = plt.figure(figsize=(10, 5))
    for num, idx in enumerate(indices_plot):
        first_img = dataset_first[idx, ...]
        second_img = dataset_second[idx, ...]
        ax = fig.add_subplot(2, 8, num + 1)
        ax.imshow(first_img)
        ax.set_title("Similarity: {0}".format(score[idx]))
        ax = fig.add_subplot(2, 8, num + 1 + 8)
        ax.imshow(second_img)
        ax.axis('off')
    plt.tight_layout()

def save_model(model_istance, model_name_save):
    if os.path.exists("Saved_model"):
        model_json = model_istance.to_json()
        with open("Saved_model/{0}.json".format(model_name_save), "w") as json_file:
            json_file.write(model_json)
        model_istance.save_weights("Saved_model/{0}.h5".format(model_name_save))
        print("Saved to disk")
    else:
        os.mkdir("Saved_model")
        model_json = model_istance.to_json()
        with open("Saved_model/{0}.json".format(model_name_save), "w") as json_file:
            json_file.write(model_json)
        model_istance.save_weights("Saved_model/{0}.h5".format(model_name_save))
        print("Saved to disk")

def predict_similarity(first_image_path, second_image_path, model_istance):
    fig = plt.figure()
    first_image = cv2.imread(first_image_path)
    second_image = cv2.imread(second_image_path)
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(first_image)
    ax.set_title("First Image")
    ax.axis('off')
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(second_image)
    ax.axis('off')
    first_image = cv2.resize(first_image, (32, 32))
    first_image = first_image / 255
    first_image = first_image.reshape(1, 32, 32, 3)
    second_image = cv2.resize(second_image, (32, 32))
    second_image = second_image / 255
    second_image = second_image.reshape(1, 32, 32, 3)
    ax.set_title("Similarity: {0}".format(model_istance.predict([first_image, second_image])))
    plt.tight_layout()

def load_model(path, file_name):
    json_file = open("{0}/{1}.json".format(path, file_name), "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("{0}/{1}.h5".format(path, file_name))
    print("Loaded from disk")
    return loaded_model

def evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test, model_istance):
    training_score = round(model_istance.evaluate([x_train[0], x_train[1]], y_train, verbose=0)[1], 4)
    validation_score = round(model_istance.evaluate([x_val[0], x_val[1]], y_val, verbose=0)[1], 4)
    test_score = round(model_istance.evaluate([x_test[0], x_test[1]], y_test, verbose=0)[1], 4)
    print("Train set accuracy is: {0}".format(training_score))
    print("Validation set accuracy is: {0}".format(validation_score))
    print("Test set accuracy is: {0}".format(test_score))

x_train, y_train = dataset_load(path="Data/train", img_res=[32, 32])
x_val, y_val = dataset_load(path="Data/val", img_res=[32, 32])
input_shape = x_train[1, ...].shape

y_train_cat = keras.utils.to_categorical(y_train, 9)
y_val_cat = keras.utils.to_categorical(y_val, 9)

print('Training dataset shape:', x_train.shape)
print('Validation dataset shape:', x_val.shape)
print(x_train.shape[0], 'training samples')
print(x_val.shape[0], 'validation samples')

first_image_train, second_image_train, mutual_score_train = generate_permutation_dataset(dataset_x=x_train,
                                                                                   dataset_y=y_train,
                                                                                   img_res=[32, 32])
first_image_val, second_image_val, mutual_score_val = generate_permutation_dataset(dataset_x=x_val,
                                                                                   dataset_y=y_val,
                                                                                   img_res=[32, 32])

plot_permutation_group(dataset_first=first_image_train,
                       dataset_second=second_image_train,
                       score=mutual_score_train)
plot_permutation_group(dataset_first=first_image_val,
                       dataset_second=second_image_val,
                       score=mutual_score_val)

img_in = Input(shape = input_shape, name = 'FeatureNet_ImageInput')
conv1 = Conv2D(32, kernel_size=3, activation='relu')(img_in)
bn1 = BatchNormalization()(conv1)
conv2 = Conv2D(32, kernel_size=3, activation='relu')(bn1)
bn2 = BatchNormalization()(conv2)
conv3 = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu')(bn2)
bn3 = BatchNormalization()(conv3)
do1 = Dropout(0.5)(bn3)
conv4 = Conv2D(64, kernel_size=3, activation='relu')(do1)
bn4 = BatchNormalization()(conv4)
conv5 = Conv2D(64, kernel_size=3, activation='relu')(bn4)
bn5 = BatchNormalization()(conv5)
conv6 = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(bn5)
bn6 = BatchNormalization()(conv6)
do2 = Dropout(0.5)(bn6)
fc = Flatten()(do2)
d1 = Dense(128, activation='relu')(fc)
bn7 = BatchNormalization()(d1)
output = Dropout(0.5)(bn7)
feature_model = Model(inputs = [img_in], outputs = [output], name = 'FeatureGenerationModel')
feature_model.summary()


img_a_in = Input(shape = input_shape, name = 'ImageA_Input')
img_b_in = Input(shape = input_shape, name = 'ImageB_Input')
img_a_feat = feature_model(img_a_in)
img_b_feat = feature_model(img_b_in)
combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
combined_features = Dense(128, activation = 'relu')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Dropout(0.5)(combined_features)
combined_features = Dense(1, activation = 'sigmoid')(combined_features)
similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')
similarity_model.summary()

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
similarity_model.compile(optimizer=adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
now = datetime.datetime.now()

checkpointer = ModelCheckpoint(filepath='logs/Saved_weights/weights.hdf5', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir="logs/{}".format("{0}{1}".format(now.hour, now.minute)))

similarity_model.fit([first_image_train, second_image_train],
                     mutual_score_train,
                     validation_data=([first_image_val, second_image_val], mutual_score_val),
                     epochs=100,
                     batch_size=16,
                     verbose=1,
                     callbacks=[tensorboard, checkpointer])

save_model(model_istance=similarity_model,
           model_name_save="siamese_similarity")

similarity_model = load_model(path="Saved_model",
                          file_name="siamese_similarity")
similarity_model.compile(optimizer=adam, loss = 'binary_crossentropy', metrics = ['mae'])


#model testing
x_test, y_test = dataset_load(path="Data/test", img_res=[32, 32])
y_test_cat = keras.utils.to_categorical(y_test, 9)

first_image_test, second_image_test, mutual_score_test = generate_permutation_dataset(dataset_x=x_test,
                                                                                      dataset_y=y_test,
                                                                                      img_res=[32, 32])

plot_permutation_group(dataset_first=first_image_arr_test,
                       dataset_second=second_image_arr_test,
                       score=mutual_score_arr_test)

predict_similarity(first_image_path="Data/test/1/1.jpg",
                   second_image_path="Data/test/1/5.png",
                   model_istance=similarity_model)

evaluate_model(x_train=[first_image_train, second_image_train],
               y_train=mutual_score_train,
               x_val=[first_image_val, second_image_val],
               y_val=mutual_score_val,
               x_test=[first_image_test, second_image_test],
               y_test=mutual_score_test,
               model_istance=similarity_model)