import numpy as np
import time

from scipy import ndimage
from tensorflow_core import keras
from matplotlib import pyplot as plt
from tensorflow_core.python.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from pathlib import Path


def save_confusion_matrix(generator, predictions, dir_name, title):
    f = open('../data/' + dir_name + '/data.txt', "a+")
    prfs = precision_recall_fscore_support(y_true=generator.classes, y_pred=predictions)
    cm = confusion_matrix(y_true=generator.classes, y_pred=predictions)
    f.write(title + ':\n cm \n' + str(cm) + '\n')
    f.write(title + ':\n prfs \n' + str(prfs) + '\n')
    f.close()


def save_test_accuracy(model, test_it, dir_name):
    score = model.evaluate(test_it)
    f = open('../data/' + dir_name + '/data.txt', "a+")
    f.write('test accuracy:' + str(score[1]) + '\n')
    f.close()


def analyze_predictions(model, train_it, test_it, dir_name):
    Path('../data/' + dir_name + '/wrong_answers').mkdir(parents=True, exist_ok=True)
    predictions = model.predict_generator(test_it, verbose=1)
    save_test_accuracy(model, test_it, dir_name)
    predictions = predictions > 0.5
    save_confusion_matrix(test_it, predictions, dir_name, 'test confusion matrix')


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
    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=5, width_shift_range=0.1,
                                                                  height_shift_range=0.05)
    train_it = data_generator.flow_from_directory('../chest_xray/train/', class_mode='binary', batch_size=1, shuffle=True)
    val_it = data_generator.flow_from_directory('../chest_xray/val/', class_mode='binary', batch_size=1, shuffle=False)
    test_it = data_generator.flow_from_directory('../chest_xray/test/', class_mode='binary', batch_size=1, shuffle=False)

    return train_it, val_it, test_it