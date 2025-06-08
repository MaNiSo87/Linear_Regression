
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
data = data.values
print(data)

Xs = data[:, 0]
ys = data[:, 1]
#print(Xs, ys)

m, n = data.shape
l = int(0.8 * m)
x_train = Xs[:l]
y_train = ys[:l]
x_test = Xs[l:]
y_test = ys[l:]
#print(x_train, y_train)

weights = []
biases = []

class Agent001():
    def __init__(self, lr= 0.001, n=10000):
        self.learning_rate = lr
        self.epoch = n
        self.weight = 0
        self.bias = 0

    def predict(self, x_train):
        return self.weight * x_train + self.bias

    def train(self, x_train, y_train):
        for i in range(1, self.epoch+1):
            y_pred = self.predict(x_train)

            y = y_pred - y_train

            dw = (1/len(x_train)) * sum((x_train * y))
            db = (1/len(x_train)) * sum(y)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 1000 == 0:
                print(f'Epoch: {i}:') 
                print(f'-Cost: {np.mean(y**2)}')
                print(f'-Weight: {self.weight}')
                weights.append(self.weight)
                print(f'-Bias: {self.bias}')
                biases.append(self.bias)



agent = Agent001()

agent.train(x_train, y_train)
predict = agent.predict(Xs)

plt.scatter(x_train, y_train, label="Train")
plt.scatter(x_test, y_test, label="Test")
plt.plot(Xs, predict, color='red', label="Predicted")
plt.legend()
plt.title("Linear Regression Result")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

plt.scatter(biases, weights, label="Train")
plt.show()
