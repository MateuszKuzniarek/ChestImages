import numpy as np
import time

from scipy import ndimage
from tensorflow_core import keras
from matplotlib import pyplot as plt
from tensorflow_core.python.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import confusion_matrix

# from tensorflow_core.python.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


# def save_image(sample, filename):
#     sample = np.asarray(sample)
#     size = sample.shape[0]
#     if sample.shape[2] == number_of_channels_for_rgb:
#         plt.imshow(sample)
#     else:
#         sample = sample.reshape(size, size)
#         plt.imshow(sample, cmap='gray')
#     plt.savefig(filename)


def save_confusion_matrix(generator, predictions, dir_name, title):
    f = open('../data/' + dir_name + '/data.txt', "a+")
    print(generator.classes)
    print(len(generator.classes))
    print(predictions)
    print(len(predictions))
    cm = confusion_matrix(y_true=generator.classes, y_pred=predictions)
    f.write(title + ':\n' + str(cm) + '\n')
    f.close()


# def save_wrong_answers(x_test, y_test, predictions, dir_name):
#     for i in range(0, len(predictions)):
#         if predictions[i] != y_test[i]:
#             filename = '../data/' + dir_name + '/wrong_answers/pred_' + \
#                        str(predictions[i]) + '_ans_' + str(y_test[i]) + '.png'
#             save_image(x_test[i], filename)


def save_test_accuracy(model, test_it, dir_name):
    score = model.evaluate(test_it)
    f = open('../data/' + dir_name + '/data.txt', "a+")
    f.write('test accuracy:' + str(score[1]) + '\n')
    f.close()


def analyze_predictions(model, train_it, test_it, dir_name):
    Path('../data/' + dir_name + '/wrong_answers').mkdir(parents=True, exist_ok=True)
    #train_predictions = model.predict_generator(train_it, verbose=1)
    #train_predictions = list(map(lambda x: np.argmax(x), train_predictions))
    #save_confusion_matrix(train_it, train_predictions, dir_name, 'train confusion matrix')
    predictions = model.predict_generator(test_it, verbose=1)
    save_test_accuracy(model, test_it, dir_name)
    #predictions = list(map(lambda x: np.argmax(x), predictions))
    save_confusion_matrix(test_it, predictions, dir_name, 'test confusion matrix')
    # save_wrong_answers(x_test, y_test, predictions, dir_name)


def start_learning(model, train_it, val_it, batch_size, epochs, dir_name):
    f = open('../data/' + dir_name + '/data.txt', "w+")
    csv_logger = CSVLogger('../data/' + dir_name + '/training.txt')
    start_time = time.time()
    history = model.fit_generator(train_it, callbacks=[csv_logger],
                                  steps_per_epoch=train_it.samples/batch_size, epochs=epochs, verbose=1,
                                  validation_data=val_it, class_weight=True)
    end_time = time.time()
    f.write('learning time: ' + str(end_time - start_time) + '\n')
    score = model.evaluate(val_it, verbose=0)
    f.write('train loss:' + str(history.history['loss'][-1]) + '\n')
    f.write('train accuracy:' + str(history.history['accuracy'][-1]) + '\n')
    f.write('val loss:' + str(score[0]) + '\n')
    f.write('val accuracy:' + str(score[1]) + '\n')
    f.close()


def prepare_data():
    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_it = data_generator.flow_from_directory('../chest_xray/train/', class_mode='binary', batch_size=1, shuffle=True)
    val_it = data_generator.flow_from_directory('../chest_xray/val/', class_mode='binary', batch_size=1, shuffle=True)
    test_it = data_generator.flow_from_directory('../chest_xray/test/', class_mode='binary', batch_size=1, shuffle=True)

    return train_it, val_it, test_it


def convert_to_three_channels(x_train, x_test):
    x_train_repeated = []
    x_test_repeated = []
    for i in range(0, len(x_train)):
        x_train_repeated.append(np.repeat(x_train[i], repeats=3, axis=-1))
    for i in range(0, len(x_test)):
        x_test_repeated.append(np.repeat(x_test[i], repeats=3, axis=-1))
    return np.asarray(x_train_repeated), np.asarray(x_test_repeated)


def rescale_images(x_train, x_test):
    x_train = np.asarray([ndimage.zoom(image, (2, 2, 1), order=1) for image in x_train])
    x_test = np.asarray([ndimage.zoom(image, (2, 2, 1), order=1) for image in x_test])
    return x_train, x_test
