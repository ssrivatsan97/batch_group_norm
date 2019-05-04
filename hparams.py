# hyperparameters

# dataset parameters
dataset_id = 2        # 1 - MNIST, 2 - Cifar10
if dataset_id == 1:
  img_size = [28,28,1]
else:
  img_size = [32,32,3]

# training parameters
learning_rate = 0.001
num_training_epochs = 20
batch_size = 50

# conv layer parameters
num_filter_layer_1 = 16
num_filter_layer_2 = 32
num_filter_layer_3 = 64
num_filter_layer_4 = 128
filter_size = 3

# fc layer parameters
num_fc1_units = 256
num_classes = 10

log_dir = './Graph'

# batch/group norm parameters
bn_fc = False
bn_conv = False
gn_fc = False
gn_conv = False
num_groups = 4