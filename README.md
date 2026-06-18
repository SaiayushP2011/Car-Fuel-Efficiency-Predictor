# Car-Fuel-Efficiency-Predictor
# Car Fuel Efficiency Predictor

A PyTorch-based ML project that predicts a car’s fuel efficiency (MPG) using factors such as horsepower, weight, cylinders, and engine size.

This project uses a custom CSV dataset to train a neural network on car specifications and estimate fuel efficiency for unseen cars. The model is trained using supervised learning and evaluated using a train/test in a 80:20 ratio split to measure how well it generalizes.

## Features

* Reads and processes CSV data
* Converts data into PyTorch tensors
* Splits data into training and testing sets
* Trains a neural network regression model
* Predicts MPG for unseen cars(test data)
* Evaluates model performance using test loss
* Prints out results of learning

## What I Learned

Through this project, I learned:

* How to work with CSV datasets in Python
* How to split data into training and testing sets
* How to build regression models using PyTorch(I improved my knowledge on this)
* How to evaluate model accuracy on unseen data
* How neural networks learn patterns from multiple input features

## Technologies Used

* Python
* PyTorch
* CSV module

## Future Improvements

* Use a larger real-world dataset
* Normalize input data for better performance
* Improve prediction accuracy
