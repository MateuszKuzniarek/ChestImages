from learning import prepare_data, start_learning, analyze_predictions
from models import get_custom_model, get_mobilenet_with_freezing
from pathlib import Path

batch_size = 32
epochs = 10

# Path("../data/test7").mkdir(parents=True, exist_ok=True)
# train_it, val_it, test_it = prepare_data()
# model = get_custom_model()
# print(train_it.samples/batch_size)
# start_learning(model, train_it, val_it, batch_size, epochs, 'test7')
# analyze_predictions(model, train_it, test_it, 'test7')

Path("../data/vgg3").mkdir(parents=True, exist_ok=True)
train_it, val_it, test_it = prepare_data()
model = get_mobilenet_with_freezing()
print(train_it.samples/batch_size)
start_learning(model, train_it, val_it, batch_size, epochs, 'vgg3')
analyze_predictions(model, train_it, test_it, 'vgg3')