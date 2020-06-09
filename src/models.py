from tensorflow_core import keras
from tensorflow_core.python.keras.models import Sequential, Model
from tensorflow_core.python.keras.layers import Dense, Dropout, Flatten
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D
from constants import num_classes, base_img_cols, base_img_rows


def get_custom_architecture():
    input_shape = (base_img_rows, base_img_cols, 3)
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_custom_model():
    model = get_custom_architecture()
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.99),
                  metrics=['accuracy'])

    return model


# def get_custom_model_2():
#     model = get_custom_architecture()
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999),
#                   metrics=['accuracy'])
#
#     return model
#
#
def _get_mobilnet():
    base_model = keras.applications.VGG19(input_shape=(base_img_rows, base_img_cols, 3),
                                                             include_top=False,
                                                             weights='imagenet')
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    final_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=final_tensor)

    return model
#
#
# def get_mobilenet_without_freezing():
#     model = _get_mobilnet()
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999),
#                   metrics=['accuracy'])
#     return model
#
#
def get_mobilenet_with_freezing():
    model = _get_mobilnet()
    model.summary()
    # print(len(model.layers))
    # for layer in model.layers:
    #     print(layer)
    # number_of_layers_to_freeze = 19
    # for i in range(0, number_of_layers_to_freeze):
    #     model.layers[i].trainable = False
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.99),
                  metrics=['accuracy'])
    return model
