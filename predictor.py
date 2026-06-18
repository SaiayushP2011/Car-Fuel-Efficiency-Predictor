import csv
import torch
import torch.nn as nn
import torch.optim as optim

x = []
y = []

with open("cars.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    
    for row in reader:
        x_part = [float(value) for value in row[0:4]]
        y_part = float(row[4])

        x.append(x_part)
        y.append(y_part)

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train = X[0:24]
y_train = y[0:24]

X_test = X[24:30]
y_test = y[24:30]

y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

epochs = 1000

class Analyser(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4,8)

        self.reLu = nn.ReLU()

        self.fc2 = nn.Linear(8,1)

    def forward(self, x):
        x = self.fc1(x)

        x = self.reLu(x)

        x = self.fc2(x)

        return x

model = Analyser()

loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.01
)
for epoch in range(epochs):
    prediction = model(X_train)
    loss = loss_function(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

test_input = X_test

with torch.no_grad():
    test_predictions = model(test_input)

print(f"prediction for {test_input}")
print(test_predictions)
print("Actual MPG:")
print(y_test)
test_loss = loss_function(test_predictions, y_test)
print(f"Test Loss: {test_loss.item()}")