import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01, dropout_rate=0.5):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)
    
    def compute_cost(self, Y, Y_hat):
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[range(m), Y])
        data_loss = np.sum(log_probs) / m
        reg_loss = (self.lambda_reg / (2 * m)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        cost = data_loss + reg_loss
        return cost
    
    def forward_propagation(self, X, training=True):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        if training:
            # Apply dropout
            D1 = np.random.rand(*A1.shape) > self.dropout_rate
            A1 *= D1
            A1 /= (1 - self.dropout_rate)
        else:
            D1 = None
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        cache = (Z1, A1, D1, Z2, A2)
        return A2, cache
    
    def backward_propagation(self, X, Y, cache):
        Z1, A1, D1, Z2, A2 = cache
        m = Y.shape[0]
        
        dZ2 = A2
        dZ2[range(m), Y] -= 1
        dZ2 /= m
        
        dW2 = np.dot(A1.T, dZ2) + (self.lambda_reg / m) * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        if D1 is not None:
            dA1 *= D1
            dA1 /= (1 - self.dropout_rate)
        dZ1 = self.relu_backward(dA1, Z1)
        
        dW1 = np.dot(X.T, dZ1) + (self.lambda_reg / m) * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, grads, learning_rate):
        dW1, db1, dW2, db2 = grads
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Generate synthetic data for demonstration
np.random.seed(42)  # For reproducibility
num_samples = 1000
num_features = 20
num_classes = 3  # Example with 3 classes

X_train = np.random.randn(num_samples, num_features)
y_train = np.random.randint(0, num_classes, size=num_samples)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64  # Example size of the hidden layer
output_size = len(np.unique(y_train))  # Assuming y_train contains integer labels
learning_rate = 0.01
lambda_reg = 0.001  # Regularization strength
dropout_rate = 0.5  # Dropout rate
num_epochs = 10  # Number of epochs for training
batch_size = 32  # Size of each mini-batch

# Initialize the neural network
nn = SimpleNN(input_size, hidden_size, output_size, lambda_reg, dropout_rate)

# Training loop
for epoch in range(num_epochs):
    # Shuffle the dataset at the beginning of each epoch
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    # Mini-batch training
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]
        
        # Forward propagation with dropout
        A2, cache = nn.forward_propagation(X_batch, training=True)
        
        # Compute the loss
        cost = nn.compute_cost(y_batch, A2)
        
        # Backward propagation to get the gradients
        grads = nn.backward_propagation(X_batch, y_batch, cache)
        
        # Update parameters
        nn.update_parameters(grads, learning_rate)
    
    print(f"Epoch {epoch + 1}, Loss: {cost}")

print("Training completed.")
