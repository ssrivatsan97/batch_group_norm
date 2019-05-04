import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import hparams as hp
import time
from tensorboardcolab import *

tbc = TensorBoardColab()

def preprocess_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    stddev = np.std(train_data, axis=0)
    
    train_data = (train_data - mean)/stddev
    test_data = (test_data - mean)/stddev
    
    return train_data, test_data

def get_data_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x = np.reshape(mnist.train.images, (mnist.train.labels.shape[0], hp.img_size[0], hp.img_size[1], hp.img_size[2]))
    train_y = mnist.train.labels
    test_x = np.reshape(mnist.test.images, (mnist.test.labels.shape[0], hp.img_size[0], hp.img_size[1], hp.img_size[2]))
    test_y = mnist.test.labels
    return (train_x, train_y, test_x, test_y)

def get_data_cifar10():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    # preprocessing the input image features
    temp1 = np.reshape(train_x, (train_x.shape[0], hp.img_size[0]*hp.img_size[1]*hp.img_size[2]))
    temp2 = np.reshape(test_x, (test_x.shape[0], hp.img_size[0]*hp.img_size[1]*hp.img_size[2]))
    
    temp1, temp2 = preprocess_data(temp1, temp2)
    
    train_x = np.reshape(temp1, (train_x.shape[0], hp.img_size[0], hp.img_size[1], hp.img_size[2]))
    test_x = np.reshape(temp2, (test_x.shape[0], hp.img_size[0], hp.img_size[1], hp.img_size[2]))
    
    # converting labels to one-hot encoding
    temp1 = np.zeros((train_y.shape[0],10), dtype=np.float32)
    temp2 = np.zeros((test_y.shape[0],10), dtype=np.float32)
    temp1[np.arange(train_y.shape[0]), np.reshape(train_y, (train_y.shape[0]))] = 1.0
    temp2[np.arange(test_y.shape[0]), np.reshape(test_y, (test_y.shape[0]))] = 1.0
    return (train_x, temp1, test_x, temp2)

def create_fc_layer(input_data, num_input_units, num_output_units, variable_scope_name, activation='linear'):
    init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    with tf.variable_scope(variable_scope_name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(name='w', shape=[num_input_units, num_output_units], initializer=init, trainable=True)
        bias = tf.get_variable(name='b', shape=[num_output_units], initializer=init, trainable=True)
    
    fc_layer_output = tf.add(tf.matmul(tf.reshape(input_data, [-1, num_input_units]), weight), bias)

#     batch normalization
    if hp.bn_fc == True and variable_scope_name != 'final_fc_layer':
        fc_layer_output = tf.contrib.layers.batch_norm(fc_layer_output)

    if activation == 'relu':
        fc_layer_output = tf.nn.relu(fc_layer_output)
    elif activation == 'softmax':
        fc_layer_output = tf.nn.softmax(fc_layer_output)
    elif activation == 'tanh':
        fc_layer_output = tf.nn.tanh(fc_layer_output)
    elif activation == 'sigmoid':
        fc_layer_output = tf.nn.sigmoid(fc_layer_output)
    else:
        pass

    return fc_layer_output

def create_conv_layer(input_data, num_input_channels, num_filters, variable_scope_name, activation='linear'):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [hp.filter_size, hp.filter_size, num_input_channels, num_filters]

    # initialise weights and bias for the filter
    init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)   # uniform = False => normal
    with tf.variable_scope(variable_scope_name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(name='w', shape=conv_filt_shape, initializer=init, trainable=True)
        bias = tf.get_variable(name='b', shape=[num_filters], initializer=init, trainable=True)

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input=input_data, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

    if variable_scope_name == 'conv_layer_2':
        for i in range(num_filters):
            tf.summary.histogram(str(i)+'before', out_layer[:,:,:,i])

    if hp.bn_conv == True:
        out_layer = tf.contrib.layers.batch_norm(out_layer)
    
    if variable_scope_name == 'conv_layer_2':
        for j in range(num_filters):
            tf.summary.histogram(str(j)+'after', out_layer[:,:,:,j])

    if activation == 'relu':
        out_layer = tf.nn.relu(out_layer)
    elif activation == 'tanh':
        out_layer = tf.nn.tanh(out_layer)
    elif activation == 'sigmoid':
        out_layer = tf.nn.sigmoid(out_layer)
    else:
        pass

    return out_layer

def create_pool_layer(input_data):
    pool_shape = [2,2]
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    pool_strides = [1, 2, 2, 1]
    output_tensor = tf.nn.max_pool(input_data, ksize=ksize, strides=pool_strides, padding='SAME')
    return output_tensor

def core_model(cnn_input):
    # create some convolutional layers
    conv_layer_1 = create_conv_layer(cnn_input, hp.img_size[2], hp.num_filter_layer_1, 'conv_layer_1', 'relu')
    pool_layer_1 = create_pool_layer(conv_layer_1)
    mod_img_size = [hp.img_size[0]//2, hp.img_size[1]//2, hp.img_size[2]]
    
    conv_layer_2 = create_conv_layer(pool_layer_1, hp.num_filter_layer_1, hp.num_filter_layer_2, 'conv_layer_2', 'relu')
    pool_layer_2 = create_pool_layer(conv_layer_2)
    mod_img_size = [mod_img_size[0]//2, mod_img_size[1]//2, mod_img_size[2]]

    conv_output = tf.reshape(pool_layer_2, [-1, mod_img_size[0]*mod_img_size[1]*hp.num_filter_layer_2])

    # setup some weights and bias values for this layer, then activate with ReLU
    fc_layer_1 = create_fc_layer(conv_output, mod_img_size[0]*mod_img_size[1]*hp.num_filter_layer_2, hp.num_fc1_units, 'first_fc_layer', 'relu')
    predicted_output = create_fc_layer(fc_layer_1, hp.num_fc1_units, hp.num_classes, 'final_fc_layer')

    return predicted_output

def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, global_norm = tf.clip_by_global_norm(t_list=gradients, clip_norm=max_gradient_norm)  # (t_list*clip_norm)/max(clip_norm, global_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", global_norm)]
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, global_norm, gradient_norm_summary

def get_next_batch(train_x, train_y, step):
    return train_x[step*hp.batch_size:(step+1)*hp.batch_size,:,:,:], train_y[step*hp.batch_size:(step+1)*hp.batch_size,:]

def train_network(train_x, train_y, test_x, test_y):
    network_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.img_size[0], hp.img_size[1], hp.img_size[2]], name="input_placeholder")
    network_output = tf.placeholder(dtype=tf.float32, shape=[None, hp.num_classes], name="label_placeholder")
#     is_training = tf.placeholder(dtype=tf.bool, name="is_training_placeholder")

    logits = core_model(network_input)

    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=network_output))
    tf.summary.scalar('cross_entropy_loss', total_loss)

    # define an accuracy assessment operation
    with tf.name_scope("accuracy"):
        prediction_comparison = tf.equal(x=tf.argmax(input=network_output, axis=1), y=tf.argmax(input=logits, axis=1))
        accuracy = tf.reduce_mean(tf.cast(x=prediction_comparison, dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # optimizer.minimize(total_loss) = optimizer.compute_gradients() -> optimizer.apply_gradients()
    optimizer = tf.train.AdamOptimizer(hp.learning_rate)

    with tf.name_scope("compute_gradients"):
        grads_and_params = optimizer.compute_gradients(loss=total_loss)
        grads, params = zip(*grads_and_params)
#         for var in grads:
#             tf.summary.histogram(var.name, var)
        # clipped_grads, global_norm, grad_norm_summary = gradient_clip(gradients=grads, max_gradient_norm=hp.max_gradient_norm)
        grads_and_vars = zip(grads, params)

    # global_step refers to the number of batches seen by the graph. Every time a batch is provided, the weights are updated in the direction 
    # that minimizes the loss. global_step just keeps track of the number of batches seen so far
    global_step = tf.train.get_or_create_global_step()
    apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads_and_params, global_step=global_step)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    train_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

    # This initializer is designed to keep the scale of the gradients roughly the same in all layers
    initializer = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)

    with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()

        train_accuracy_list = []
        test_accuracy_list = []
        train_loss_list = []
        iteration_list = []

        num_train_samples = train_y.shape[0]
        num_batches = int(num_train_samples/hp.batch_size)
        print("Training Started")
        start_time = time.time()

        for epoch in range(hp.num_training_epochs):
            print('Epochs :', epoch)
            for itr in range(num_batches):
                [batch_x, batch_y] = get_next_batch(train_x, train_y, itr)
                _, loss_val, summary, iteration_count = sess.run([apply_gradient_op, total_loss, merged_summary, global_step], 
                    feed_dict={network_input:batch_x, network_output:batch_y})
                train_loss_list.append(loss_val)
                
                train_writer.add_summary(summary, iteration_count)
                if iteration_count%10 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={network_input:train_x[0:20000], network_output:train_y[0:20000]})
                    test_accuracy = sess.run(accuracy, feed_dict={network_input:test_x, network_output:test_y})
                    train_accuracy_list.append(train_accuracy)
                    test_accuracy_list.append(test_accuracy)
                    iteration_list.append(iteration_count)
#                     print("Epoch :", epoch, "iteration :", iteration_count)
            
            lst = list(range(num_train_samples))
            np.random.shuffle(lst)
            train_x = train_x[lst]
            train_y = train_y[lst]

        print("Training Time :", (time.time() - start_time))
        
        train_accuracy_list = np.array(train_accuracy_list)
        test_accuracy_list = np.array(test_accuracy_list)
        train_loss_list = np.array(train_loss_list)
        iteration_list = np.array(iteration_list)

        start_time = time.time()
        print("final training accuracy", sess.run(accuracy, feed_dict={network_input:train_x, network_output:train_y}))
        print("Inference time in training data :", time.time() - start_time)
        start_time = time.time()
        print("final test accuracy", sess.run(accuracy, feed_dict={network_input:test_x, network_output:test_y}))
        print("Inference Time on test data :", time.time() - start_time)
        
        if hp.bn_conv == True:
            np.save('train_acc_std_bn.npy', train_accuracy_list)
            np.save('test_acc_std_bn.npy', test_accuracy_list)
            np.save('train_loss_std_bn.npy', train_loss_list)
            np.save('iter_list.npy', iteration_list)
        else:
            np.save('train_acc_std.npy', train_accuracy_list)
            np.save('test_acc_std.npy', test_accuracy_list)
            np.save('train_loss_std.npy', train_loss_list)
            np.save('iter_list.npy', iteration_list)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_data_mnist() if hp.dataset_id == 1 else get_data_cifar10()
    train_network(train_x, train_y, test_x, test_y)