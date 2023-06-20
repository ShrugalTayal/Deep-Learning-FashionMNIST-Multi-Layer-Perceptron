# FashionMNIST--MLP-implementation-from-scratch
FMNIST- MLP implementation from scratch using Numpy, Pandas and Matplotlib

```markdown
## Neural Network Implementation

### Modules Imported
- `numpy` as np
- `pandas` as pd
- `matplotlib.cm` as cm
- `matplotlib.pyplot` as plt
- `math.log` from math
- `random`
- `pickle`

### Data Loading from Google Drive
- Commented out the `drive.mount()` command since it's specific to Google Colab
- Set the path for the training data file
- Loaded the training data using `pd.read_csv`
- Checked the shape of the training data (60000, 785)
- Set the path for the test data file
- Loaded the test data using `pd.read_csv`
- Checked the shape of the test data (10000, 785)

### Data Visualization
- Extracted the labels and image data from the training data
- Randomly selected an object from the training data
- Plotted the image and its corresponding label using `matplotlib`

```python
plt.title((vis_train_labels[object]))
plt.imshow(vis_train_data[object].reshape(28,28), cmap=cm.binary)
```

- Extracted the labels and image data from the test data
- Randomly selected an object from the test data
- Plotted the image and its corresponding label using `matplotlib`

```python
plt.title((vis_test_labels[object]))
plt.imshow(vis_test_data[object].reshape(28,28), cmap=cm.binary)
```

### Data Preprocessing
- Defined a function `prepare_train_data` to preprocess the training data
- Shuffled the training data using `np.random.shuffle`
- Transposed the data for further processing
- Normalized the image data by dividing it by 255
- Extracted the labels and features from the preprocessed data

```python
Y_train, X_train = prepare_train_data(train_data)
```

- Defined a function `prepare_test_data` to preprocess the test data
- Transposed the data for further processing
- Normalized the image data by dividing it by 255
- Extracted the labels and features from the preprocessed data

```python
Y_test, X_test = prepare_test_data(test_data)
```

### Activation Functions
- Defined the sigmoid function and its derivative
- Defined the ReLU function and its derivative
- Defined the softmax function

### Forward Propagation
- Defined the forward propagation function for the sigmoid activation function
- Calculated the weighted sum and activation outputs for each layer

```python
Z1, A1, Z2, A2 = forward_prop_Sigmoid(W1, b1, W2, b2, X)
```

- Defined the forward propagation function for the ReLU activation function
- Calculated the weighted sum and activation outputs for each layer

```python
Z1, A1, Z2, A2 = forward_prop_ReLU(W1, b1, W2, b2, X)
```

### Backward Propagation
- Defined the backward propagation function for the sigmoid activation function
- Calculated the gradients of the weights and biases

```python
dW1, db1, dW2, db2 = backward_prop_Sigmoid(Z1, A1, Z2, A2, W1, W2, X, Y, Ymax)
```

- Defined the backward propagation function for the ReLU activation function
- Calculated the gradients of the weights and biases

```python
dW1, db1, dW2, db2 = backward_prop_ReLU(Z1, A1, Z2, A2, W1, W2, X, Y, Ymax)
```
