from learning import prepare_data, start_learning, analyze_predictions, get_vgg19
from pathlib import Path

batch_size = 32
epochs = 10

Path("../data/vgg").mkdir(parents=True, exist_ok=True)
train_it, val_it, test_it = prepare_data()
model = get_vgg19()
print(train_it.samples/batch_size)
start_learning(model, train_it, val_it, batch_size, epochs, 'vgg')
analyze_predictions(model, train_it, test_it, 'vgg')