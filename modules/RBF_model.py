import numpy as np


class RBF:
    def __init__(self, num_centers):
        self.num_centers = num_centers
        np.random.seed(42)
        self.weights = np.random.randn(self.num_centers)
        self.bias = 1
        
    def set_data(self, training_data, centers, standard_deviation):
        self.training_inputs = training_data[0]
        self.training_outputs = training_data[1]
        self.centers = centers
        self.standard_deviation = standard_deviation
        
    def forward(self, input_data):
        self.phi = self.activation(input_data, self.centers, self.standard_deviation)
        output = np.dot(self.weights, self.phi) + self.bias
        return output
    
    def backward(self, input_data, target_output, predicted_output):
        loss = self.mean_squared_error(target_output, predicted_output)
        delta = target_output - predicted_output
        gradient_weights = delta * self.phi
        gradient_bias = delta
        gradient_centers = delta * np.dot(self.weights, self.phi) * ((input_data - self.centers) / (self.standard_deviation ** 2))
        gradient_std_dev = delta * np.dot(self.weights, self.phi) * (np.linalg.norm(input_data - self.centers, axis=1) ** 2) / (self.standard_deviation ** 3)
        return loss, gradient_weights, gradient_bias, gradient_centers, gradient_std_dev
        
    def train(self, num_epochs, learning_rate):
        self.loss_history = []
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            for i in range(self.training_inputs.shape[0]):
                predicted_output = self.forward(self.training_inputs[i])
                loss, gradient_weights, gradient_bias, gradient_centers, gradient_std_dev = self.backward(self.training_inputs[i], self.training_outputs[i], predicted_output)
                self.weights += learning_rate * gradient_weights
                self.bias += learning_rate * gradient_bias
                self.centers += learning_rate * gradient_centers
                self.standard_deviation += learning_rate * gradient_std_dev
                total_loss += ((self.training_outputs[i] - predicted_output) ** 2) / 2
            total_loss /= self.training_inputs.shape[0]
            self.loss_history.append(total_loss)

            # print(f"Epoch {epoch}, Loss: {total_loss}")
            training_info = {
                'epoch': epoch,
                'loss': total_loss
            }
            yield training_info  # 使用生成器返回每个 epoch 的信息

    def euclidean_distance(self, data_point, center_point):
        return np.linalg.norm(data_point - center_point, axis=1)

    def activation(self, input_data, centers, standard_deviation):
        return np.exp(-1 / (2 * (standard_deviation ** 2)) * (self.euclidean_distance(centers, input_data) ** 2))
    
    def mean_squared_error(self, target_output, predicted_output):
        return ((target_output - predicted_output) ** 2) / 2
