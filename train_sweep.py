import wandb
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

default_parameters = dict(
    n_layers= 3,
    hidden_layer_size= 64,
    learn_rate= 1e-3,
    batch_size = 16,
    epochs=10,
    alpha = 0.0005,
    optimizer = "gradient_descent",
    activation = "sigmoid",
    weight_init = "random"
    )

run = wandb.init(config= default_parameters,project="cs6910_assignment_1", entity="neilghosh")
config = wandb.config

#import the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#store the image class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plot one sample image from each class
sample_images = []
plt.figure(figsize=(10,5))
for i in range(len(class_names)):
    sample_images.append(train_labels.tolist().index(i))
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[sample_images[i]], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[sample_images[i]]])


# convert integer to one-hot vector
def one_hot_vector(x):
    y = np.zeros((x.size, x.max()+1))
    y[np.arange(x.size),x] = 1
    return y

def activation_func(z, activation="sigmoid"):
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif activation == "softmax":
        z1 = np.exp(z)
        z2 = z1.sum(axis=1)
        z2 = np.dstack([z2])
        return z1/z2
    elif activation == "tanh":
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif activation == "relu":
        return np.maximum(0, z)

    else:
        return "Error"

def activation_derivative(z, activation="sigmoid"):
    if activation == "sigmoid":
      sig = 1 / (1 + np.exp(-z))
      return sig*(1-sig)
    elif activation == "tanh":
      tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
      return 1 - tanh**2
    elif activation == "relu":
      relu = np.maximum(0, z)
      relu[relu > 0] = 1
      return relu
    else:
      return "Error"

def data_augment(train_images,train_labels)
    datagen = ImageDataGenerator(rotation_range=45,width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='reflect')

    sep_data={}
    sep_data_created={}
    for i in range(10):
        sep_data[i]=list()
        sep_data_created[i]=list()

    for i in range(train_images.shape[0]):
        j = np.dstack([train_images[i,:,:]])
        sep_data[train_labels[i]].append(j)

    for label in range(10):
        sep_data[label]=np.array(sep_data[label])
        i=0
        for batch in datagen.flow(sep_data[label],batch_size=32):
            l = batch.tolist()
            sep_data_created[label].extend(l)
            i=i+1
            if(i==100):
                break

    for label in range(10):
        temp=np.array(sep_data_created[label])
        sep_data[label]=np.concatenate((sep_data[label],temp),axis=0)    
        sep_data[label]=sep_data[label].reshape(sep_data[label].shape[0],sep_data[label].shape[1],sep_data[label].shape[2])

    train_temp=sep_data[9]
    train_temp_label=9*np.ones(sep_data[9].shape[0])

    for label in range(9):
        train_temp = np.concatenate((train_temp,sep_data[label]),axis=0)
        train_temp_label = np.concatenate((train_temp_label,label*np.ones(sep_data[label].shape[0])),axis=0)

    return train_temp,train_temp_labels.astype(int)

#train_images, train_labels = data_augment(trainn_images, train_labels) removing data augmentation


input_size = train_images.shape[1]*train_images.shape[2]
output_size = len(class_names)

train_x = train_images.reshape([train_images.shape[0], input_size, 1])/255
test_x = test_images.reshape([test_images.shape[0], input_size, 1])/255

# shuffle the training data
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)

train_y = one_hot_vector(train_labels)
test_y = one_hot_vector(test_labels)

train_x = train_x[indices]
train_y = train_y[indices]

train_y = np.dstack([train_y])
test_y = np.dstack([test_y])

val_x = train_x[int(len(train_x)*0.9):,:,:]
val_y = train_y[int(len(train_y)*0.9):,:,:]

train_x = train_x[:int(len(train_x)*0.9),:,:]
train_y = train_y[:int(len(train_y)*0.9),:,:]


nn_layers = [{"input_dim" : input_size, "output_dim" : config.hidden_layer_size, "activation" : config.activation}]
for i in range(config.n_layers-1):
  nn_layers.append({"input_dim" : config.hidden_layer_size, "output_dim" : config.hidden_layer_size, "activation" : config.activation})
nn_layers.append({"input_dim" : config.hidden_layer_size, "output_dim" : output_size, "activation" : "softmax"})


def init_layers(nn_layers, weight_init, seed = 45):
    np.random.seed(seed)
    weights = {}

    for i, layer in enumerate(nn_layers):
        layer_no = i + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        if weight_init == "random":
          weights['w' + str(layer_no)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
          weights['b' + str(layer_no)] = np.random.randn(layer_output_size, 1) * 0.1
        elif weight_init == "xavier":
          limit = np.sqrt(6/(layer_input_size + layer_output_size))
          weights['w' + str(layer_no)] = np.random.uniform(-limit, limit, size=(layer_output_size, layer_input_size))
          weights['b' + str(layer_no)] = np.random.uniform(-limit, limit, size=(layer_output_size, 1))
        else:
          return "Error - Weight Initialization"

    return weights


def forward_prop(train_x, weights, nn_layers):
    layer_output = {}
    h_curr = train_x

    layer_output["h0"] = train_x
    layer_output["a0"] = train_x

    for i, layer in enumerate(nn_layers):
        layer_no = i + 1
        h_prev = h_curr

        activation = layer["activation"]
        w_curr = weights["w" + str(layer_no)]
        b_curr = weights["b" + str(layer_no)]

        a_curr = np.matmul(w_curr, h_prev) + b_curr
        h_curr = activation_func(a_curr, activation)

        layer_output["a" + str(layer_no)] = a_curr
        layer_output["h" + str(layer_no)] = h_curr

    return h_curr, layer_output


# cross entropy loss + regularization

def loss_func(y_hat, y, alpha, weights, nn_layers):
    cross_entropy = -np.multiply(y, np.log(y_hat)).sum() / len(y_hat)

    l2_reg = 0
    for i, layer in enumerate(nn_layers):
        layer_no = i + 1
        l2_reg += np.sum(weights["w" + str(layer_no)] ** 2)

    l2_reg = (alpha / 2) * l2_reg / len(y_hat)

    return cross_entropy + l2_reg
# accuracy
def accuracy_func(y_hat, train_y):
    correct_pred = np.argmax(y_hat, axis = 1) == np.argmax(train_y, axis = 1)
    return sum(bool(x) for x in correct_pred)/len(y_hat)


def back_prop(y_hat, y, layer_output, weights, nn_layers, alpha):
    gradients = {}

    da_prev = -(y - y_hat)

    m = len(y_hat)

    for i, layer in reversed(list(enumerate(nn_layers))):
        layer_index = i + 1

        da_curr = da_prev

        a_prev = layer_output["a" + str(i)]
        h_prev = layer_output["h" + str(i)]
        dh_prev = np.zeros(h_prev.shape)

        a_curr = layer_output["a" + str(layer_index)]
        w_curr = weights["w" + str(layer_index)]
        b_curr = weights["b" + str(layer_index)]
        dw_curr = np.zeros(w_curr.shape)
        db_curr = np.zeros(b_curr.shape)

        for j in range(m):
            dw_curr += np.dot(da_curr[j], h_prev[j].T)
            db_curr += da_curr[j]
            dh_prev[j] = np.dot(w_curr.T, da_curr[j])

        dw_curr += alpha * w_curr

        dw_curr /= m
        db_curr /= m

        if i > 0:
            activation = nn_layers[i - 1]["activation"]
            da_prev = np.multiply(dh_prev, activation_derivative(a_prev, activation))

        gradients["dw" + str(layer_index)] = dw_curr
        gradients["db" + str(layer_index)] = db_curr

    return gradients


def calculate_loss_accuracy(train_x, train_y, test_x, test_y, weights, nn_layers, alpha):

    y_hat, layer_output = forward_prop(train_x, weights, nn_layers)
    training_loss = (loss_func(y_hat, train_y, alpha, weights, nn_layers))
    training_accuracy = (accuracy_func(y_hat, train_y))

    y_hat, layer_output = forward_prop(test_x, weights, nn_layers)
    val_loss = (loss_func(y_hat, test_y, alpha, weights, nn_layers))
    val_accuracy = (accuracy_func(y_hat, test_y))

    wandb.log({"train_loss": training_loss, "train_accuracy": training_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

    return training_loss, training_accuracy, val_loss, val_accuracy


def gradient_descent(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):

  training_loss_list = []
  training_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []

  batch_x = np.array(np.array_split(train_x, n_batches))
  batch_y = np.array(np.array_split(train_y, n_batches))

  for i in range(epochs):
    for j in range(n_batches):

        y_hat, layer_output = forward_prop(batch_x[j], weights, nn_layers)
        gradients = back_prop(y_hat, batch_y[j], layer_output, weights, nn_layers, alpha)

        for k, layer in enumerate(nn_layers):
          weights["w" + str(k+1)] -= eta * gradients["dw" + str(k+1)]
          weights["b" + str(k+1)] -= eta * gradients["db" + str(k+1)]

    training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x, train_y, test_x, test_y, weights, nn_layers, alpha)

    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

    print((str(i+1)) + "/" + str(epochs) + " epochs completed")

  return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


def momentum_gd(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):

  training_loss_list = []
  training_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []

  batch_x = np.array(np.array_split(train_x, n_batches))
  batch_y = np.array(np.array_split(train_y, n_batches))

  prev_weights = {}

  gamma = 0.9

  for k, layer in enumerate(nn_layers):
          prev_weights["w" + str(k+1)] = np.zeros(weights["w" + str(k+1)].shape)
          prev_weights["b" + str(k+1)] = np.zeros(weights["b" + str(k+1)].shape)

  for i in range(epochs):
    for j in range(n_batches):

        y_hat, layer_output = forward_prop(batch_x[j], weights, nn_layers)
        gradients = back_prop(y_hat, batch_y[j], layer_output, weights, nn_layers, alpha)

        for k, layer in enumerate(nn_layers):
          prev_weights["w" + str(k+1)] = gamma * prev_weights["w" + str(k+1)] + eta * gradients["dw" + str(k+1)]
          prev_weights["b" + str(k+1)] = gamma * prev_weights["b" + str(k+1)] + eta * gradients["db" + str(k+1)]

          weights["w" + str(k+1)] -= prev_weights["w" + str(k+1)]
          weights["b" + str(k+1)] -= prev_weights["b" + str(k+1)]


    training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x, train_y, test_x, test_y, weights, nn_layers, alpha)

    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

    print((str(i+1)) + "/" + str(epochs) + " completed")

  return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


def nesterov_gd(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):
    training_loss_list = []
    training_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    batch_x = np.array(np.array_split(train_x, n_batches))
    batch_y = np.array(np.array_split(train_y, n_batches))

    prev_weights = {}
    look_ahead_w = {}

    gamma = 0.9

    for k, layer in enumerate(nn_layers):
        prev_weights["w" + str(k + 1)] = np.zeros(weights["w" + str(k + 1)].shape)
        prev_weights["b" + str(k + 1)] = np.zeros(weights["b" + str(k + 1)].shape)

    for i in range(epochs):
        for j in range(n_batches):

            for k, layer in enumerate(nn_layers):
                look_ahead_w["w" + str(k + 1)] = weights["w" + str(k + 1)] - gamma * prev_weights["w" + str(k + 1)]
                look_ahead_w["b" + str(k + 1)] = weights["b" + str(k + 1)] - gamma * prev_weights["b" + str(k + 1)]

            y_hat, layer_output = forward_prop(batch_x[j], look_ahead_w, nn_layers)
            gradients = back_prop(y_hat, batch_y[j], layer_output, look_ahead_w, nn_layers, alpha)

            for k, layer in enumerate(nn_layers):
                prev_weights["w" + str(k + 1)] = gamma * prev_weights["w" + str(k + 1)] + eta * gradients[
                    "dw" + str(k + 1)]
                prev_weights["b" + str(k + 1)] = gamma * prev_weights["b" + str(k + 1)] + eta * gradients[
                    "db" + str(k + 1)]

                weights["w" + str(k + 1)] -= prev_weights["w" + str(k + 1)]
                weights["b" + str(k + 1)] -= prev_weights["b" + str(k + 1)]

        training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x,
                                                                                                         train_y,
                                                                                                         test_x, test_y,
                                                                                                         weights,
                                                                                                         nn_layers, alpha)

        training_loss_list.append(training_loss)
        training_accuracy_list.append(training_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        print((str(i + 1)) + "/" + str(epochs) + " completed")

    return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


def rmsprop(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):

  training_loss_list = []
  training_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []

  batch_x = np.array(np.array_split(train_x, n_batches))
  batch_y = np.array(np.array_split(train_y, n_batches))

  v = {}

  beta = 0.9
  epsilon = 1e-8

  for k, layer in enumerate(nn_layers):
          v["w" + str(k+1)] = np.zeros(weights["w" + str(k+1)].shape)
          v["b" + str(k+1)] = np.zeros(weights["b" + str(k+1)].shape)

  for i in range(epochs):
    for j in range(n_batches):

        y_hat, layer_output = forward_prop(batch_x[j], weights, nn_layers)
        gradients = back_prop(y_hat, batch_y[j], layer_output, weights, nn_layers, alpha)

        for k, layer in enumerate(nn_layers):
          v["w" + str(k+1)] = beta * v["w" + str(k+1)] + (1-beta) * gradients["dw" + str(k+1)]**2
          v["b" + str(k+1)] = beta * v["b" + str(k+1)] + (1-beta) * gradients["db" + str(k+1)]**2

          weights["w" + str(k+1)] -= eta * np.divide(gradients["dw" + str(k+1)], np.sqrt(v["w" + str(k+1)] + epsilon))
          weights["b" + str(k+1)] -= eta * np.divide(gradients["db" + str(k+1)], np.sqrt(v["b" + str(k+1)] + epsilon))


    training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x, train_y, test_x, test_y, weights, nn_layers, alpha)

    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

    print((str(i+1)) + "/" + str(epochs) + " completed")

  return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list

def adam(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):

  training_loss_list = []
  training_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []

  batch_x = np.array(np.array_split(train_x, n_batches))
  batch_y = np.array(np.array_split(train_y, n_batches))

  v = {}
  v_hat = {}
  m = {}
  m_hat = {}

  beta1 = 0.9
  beta2 = 0.999
  epsilon = 1e-8

  for k, layer in enumerate(nn_layers):
          v["w" + str(k+1)] = np.zeros(weights["w" + str(k+1)].shape)
          v["b" + str(k+1)] = np.zeros(weights["b" + str(k+1)].shape)
          m["w" + str(k+1)] = np.zeros(weights["w" + str(k+1)].shape)
          m["b" + str(k+1)] = np.zeros(weights["b" + str(k+1)].shape)

  t = 0

  for i in range(epochs):
    for j in range(n_batches):

        t += 1

        y_hat, layer_output = forward_prop(batch_x[j], weights, nn_layers)
        gradients = back_prop(y_hat, batch_y[j], layer_output, weights, nn_layers, alpha)

        for k, layer in enumerate(nn_layers):
          v["w" + str(k+1)] = beta2 * v["w" + str(k+1)] + (1-beta2) * gradients["dw" + str(k+1)]**2
          v["b" + str(k+1)] = beta2 * v["b" + str(k+1)] + (1-beta2) * gradients["db" + str(k+1)]**2

          m["w" + str(k+1)] = beta1 * m["w" + str(k+1)] + (1-beta1) * gradients["dw" + str(k+1)]
          m["b" + str(k+1)] = beta1 * m["b" + str(k+1)] + (1-beta1) * gradients["db" + str(k+1)]

          v_hat["w" + str(k+1)] = np.divide(v["w" + str(k+1)], (1-beta2**t))
          v_hat["b" + str(k+1)] = np.divide(v["b" + str(k+1)], (1-beta2**t))

          m_hat["w" + str(k+1)] = np.divide(m["w" + str(k+1)], (1-beta1**t))
          m_hat["b" + str(k+1)] = np.divide(m["b" + str(k+1)], (1-beta1**t))

          weights["w" + str(k+1)] -= eta * np.divide(m_hat["w" + str(k+1)], np.sqrt(v_hat["w" + str(k+1)] + epsilon))
          weights["b" + str(k+1)] -= eta * np.divide(m_hat["b" + str(k+1)], np.sqrt(v_hat["b" + str(k+1)] + epsilon))


    training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x, train_y, test_x, test_y, weights, nn_layers, alpha)

    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

    print((str(i+1)) + "/" + str(epochs) + " completed")

  return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


def nadam(train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha):
    training_loss_list = []
    training_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    batch_x = np.array(np.array_split(train_x, n_batches))
    batch_y = np.array(np.array_split(train_y, n_batches))

    v = {}
    v_hat = {}
    m = {}
    m_hat = {}

    look_ahead_w = {}
    look_ahead_m_hat = {}
    look_ahead_v_hat = {}

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for k, layer in enumerate(nn_layers):
        v["w" + str(k + 1)] = np.zeros(weights["w" + str(k + 1)].shape)
        v["b" + str(k + 1)] = np.zeros(weights["b" + str(k + 1)].shape)
        m["w" + str(k + 1)] = np.zeros(weights["w" + str(k + 1)].shape)
        m["b" + str(k + 1)] = np.zeros(weights["b" + str(k + 1)].shape)

    t = 0

    for i in range(epochs):
        for j in range(n_batches):

            t += 1

            for k, layer in enumerate(nn_layers):
                look_ahead_v_hat["w" + str(k + 1)] = np.divide(beta2 * v["w" + str(k + 1)], (1 - beta2 ** t))
                look_ahead_v_hat["b" + str(k + 1)] = np.divide(beta2 * v["b" + str(k + 1)], (1 - beta2 ** t))

                look_ahead_m_hat["w" + str(k + 1)] = np.divide(beta1 * m["w" + str(k + 1)], (1 - beta1 ** t))
                look_ahead_m_hat["b" + str(k + 1)] = np.divide(beta1 * m["b" + str(k + 1)], (1 - beta1 ** t))

                look_ahead_w["w" + str(k + 1)] = weights["w" + str(k + 1)] - eta * np.divide(
                    look_ahead_m_hat["w" + str(k + 1)], np.sqrt(look_ahead_v_hat["w" + str(k + 1)] + epsilon))
                look_ahead_w["b" + str(k + 1)] = weights["b" + str(k + 1)] - eta * np.divide(
                    look_ahead_m_hat["b" + str(k + 1)], np.sqrt(look_ahead_v_hat["b" + str(k + 1)] + epsilon))

            y_hat, layer_output = forward_prop(batch_x[j], look_ahead_w, nn_layers)
            gradients = back_prop(y_hat, batch_y[j], layer_output, look_ahead_w, nn_layers, alpha)

            for k, layer in enumerate(nn_layers):
                v["w" + str(k + 1)] = beta2 * v["w" + str(k + 1)] + (1 - beta2) * gradients["dw" + str(k + 1)] ** 2
                v["b" + str(k + 1)] = beta2 * v["b" + str(k + 1)] + (1 - beta2) * gradients["db" + str(k + 1)] ** 2

                m["w" + str(k + 1)] = beta1 * m["w" + str(k + 1)] + (1 - beta1) * gradients["dw" + str(k + 1)]
                m["b" + str(k + 1)] = beta1 * m["b" + str(k + 1)] + (1 - beta1) * gradients["db" + str(k + 1)]

                v_hat["w" + str(k + 1)] = np.divide(v["w" + str(k + 1)], (1 - beta2 ** t))
                v_hat["b" + str(k + 1)] = np.divide(v["b" + str(k + 1)], (1 - beta2 ** t))

                m_hat["w" + str(k + 1)] = np.divide(m["w" + str(k + 1)], (1 - beta1 ** t))
                m_hat["b" + str(k + 1)] = np.divide(m["b" + str(k + 1)], (1 - beta1 ** t))

                weights["w" + str(k + 1)] -= eta * np.divide(m_hat["w" + str(k + 1)],
                                                             np.sqrt(v_hat["w" + str(k + 1)] + epsilon))
                weights["b" + str(k + 1)] -= eta * np.divide(m_hat["b" + str(k + 1)],
                                                             np.sqrt(v_hat["b" + str(k + 1)] + epsilon))

        training_loss, training_accuracy, val_loss, val_accuracy = calculate_loss_accuracy(train_x,
                                                                                                         train_y,
                                                                                                         test_x, test_y,
                                                                                                         weights,
                                                                                                         nn_layers, alpha)

        training_loss_list.append(training_loss)
        training_accuracy_list.append(training_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        print((str(i + 1)) + "/" + str(epochs) + " completed")

    return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


def train(train_x, train_y, test_x, test_y, nn_layers, epochs, eta, batch_size, optimizer, weight_init, alpha):
    weights = init_layers(nn_layers, weight_init)

    n_batches = len(train_x) // batch_size

    if optimizer == "gradient_descent":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = gradient_descent(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    elif optimizer == "momentum_gradient_descent":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = momentum_gd(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    elif optimizer == "nesterov_accelerated_gradient_descent":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = nesterov_gd(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    elif optimizer == "rmsprop":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = rmsprop(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    elif optimizer == "adam":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = adam(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    elif optimizer == "nadam":
        weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list = nadam(
            train_x, train_y, test_x, test_y, weights, nn_layers, eta, epochs, n_batches, alpha)

    else:
        return "Error - Wrong Optimizer"

    return weights, training_loss_list, training_accuracy_list, val_loss_list, val_accuracy_list


eta = config.learn_rate
epochs = config.epochs
batch_size = config.batch_size
optimizer = config.optimizer
weight_init = config.weight_init
alpha = config.alpha

# batch_size = 1 for stochastic and batch_size = len(train_x) for batch updates
#available_optimizers = ["gradient_descent", "momentum_gradient_descent", "nesterov_accelerated_gradient_descent", "rmsprop", "adam", "nadam"]

weights, train_loss, train_accuracy, val_loss, val_accuracy = train(train_x, train_y, val_x, val_y, nn_layers, epochs, eta, batch_size, optimizer, weight_init, alpha)

# save the optimized weights
with open('weights.pickle', 'wb') as handle:
    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load previously saved weights
with open('weights.pickle', 'rb') as handle:
    weights = pickle.load(handle)

# wandb logging

#wandb.log({"sample_images": plt})
# wandb.log({"epochs": epochs, "batch_size": batch_size, "learning_rate" : eta, "optimizer" : optimizer, "weight_init" : weight_init, "alpha" : alpha})