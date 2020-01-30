#!/usr/bin/python
# 
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/20/2020
# Please use this template as a starting point, 
# to support autograding with Gradescope.
import argparse
import numpy as np


# Process arguments for k-NN classification
def handle_args():
    parser = argparse.ArgumentParser(description=
                 'Make predictions using the k-NN algorithms.')

    parser.add_argument('-k', type=int, default=1, help='Number of nearest neighbors to consider')
    parser.add_argument('--varnorm', action='store_true', help='Normalize features to zero mean and unit variance')
    parser.add_argument('--rangenorm', action='store_true', help='Normalize features to the range [-1,+1]')
    parser.add_argument('--exnorm', action='store_true', help='Normalize examples to unit length')
    parser.add_argument('train',  help='Training data file')
    parser.add_argument('test',   help='Test data file')

    return parser.parse_args()


# Load data from a file
def read_data(filename):
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  x = data[:, 0:-1]
  y = data[:, -1]
  return (x,y)


# Distance between instances x1 and x2
def dist(x1, x2):
  # TODO: YOUR CODE HERE
  distance = x1 - x2
  return np.einsum('i, i->', *2*(distance,))
  
# Predict label for instance x, using k nearest neighbors in training data
def classify(train_x, train_y, k, x):
  # TODO: YOUR CODE HERE
  # keep a list of all the distances with the values
  distances = []
  for i, data in enumerate(train_x):
    # caluclate distance between x and data
    distance = dist(x, data)
    # add distance and label to list
    distances.append((distance, train_y[i]))

  # sort list of distances
  distances.sort()

  # tally up the results of the first k items in the sorted list
  results = {}
  best = (None, 0)
  for i in range(k):
    distance, label = distances[i]
    # increment label by 1 if the label exists, or set to 1 if it doesn't
    results[label] = results.get(label, 0) + 1
    # keep track of the label with the most "votes"
    if (results[label] > best[1]):
      best = (label, results[label])

  return best[0]

# Process the data to normalize features and/or examples.
# NOTE: You need to normalize both train and test data the same way.
def normalize_data(train_x, test_x, rangenorm, varnorm, exnorm):
  train = np.copy(train_x)
  test = np.copy(test_x)

  if rangenorm:
    print("rangenorm")  
  if varnorm:
    print("varnorm")  
  if exnorm:
    print("exnorm")
  with np.printoptions(threshold=np.inf):
    print("train")
    print(train)
    print("test")
    print(test)
  print('-'*10)

  if rangenorm:
    '''
    Feature range normalization should rescale instances 
    to range from -1 (mini- mum) to +1 (maximum), 
    according to values in the training data
    '''

    # find min and max
    train_min = np.min(train_x, axis=0)
    train_max = np.max(train_x, axis=0)

    # normalize training values
    # [(x - min) / (max - min)] * 2 - 1
    for i in range(len(train)):
      for j in range(len(train[i])):
        train[i][j] = ((train[i][j] - train_min[j]) / (train_max[j] - train_min[j])) * 2 - 1

    for i in range(len(test)):
      for j in range(len(test[i])):
        test[i][j] = ((test[i][j] - train_min[j]) / (train_max[j] - train_min[j])) * 2 - 1

  if varnorm:
    '''
    Feature variance normalization should rescale instances 
    so they have a standard deviation of 1 in the training data.
    '''
    # find mean
    train_xbar = np.mean(train_x, axis=0)
    # find std
    train_std = np.std(train_x, axis=0)
    
    # normalize
    # (x - xbar) / std
    for i in range(len(train)):
      for j in range(len(train[i])):
        train[i][j] = (train[i][j] - train_xbar[j]) / train_std[j]

    for i in range(len(test)):
      for j in range(len(test[i])):
        test[i][j] = (test[i][j] - train_xbar[j]) / train_std[j]

  if exnorm:
    '''
    Example magnitude normalization should rescale 
    each example to have a magnitude of 1 (under a Euclidean norm).
    '''
    # TODO: YOUR CODE HERE
    # find euclidean norm
    norm_x = np.linalg.norm(train_x.astype(float), axis=0)

    # normalize
    # x / ||x|| 
    for i in range(len(train)):
      for j in range(len(train[i])):
        if norm_x[j] == 0:
          train[i][j] = 0
        else:
          train[i][j] = float(train[i][j]) / norm_x[j]

    for i in range(len(test)):
      for j in range(len(test[i])):
        if norm_x[j] == 0:
          test[i][j] = 0
        else:
          test[i][j] = test[i][j] / norm_x[j]


  train = np.nan_to_num(train)
  test = np.nan_to_num(test)

  with np.printoptions(threshold=np.inf):
    print("train")
    print(train)
    print("test")
    print(test)
    print('-'*20)
    print()

  return train, test

# Run classifier and compute accuracy
def runTest(test_x, test_y, train_x, train_y, k):
  correct = 0
  i = 1
  for (x,y) in zip(test_x, test_y):
    i += 1
    if classify(train_x, train_y, k, x) == y:
      correct += 1
  acc = float(correct)/len(test_x)
  return acc

# Load train and test data.  Learn model.  Report accuracy.
# (NOTE: You shouldn't need to change this.)
def main():

  args = handle_args()

  # Read in lists of examples.  Each example is a list of attribute values,
  # where the last element in the list is the class value.
  (train_x, train_y) = read_data(args.train)
  (test_x, test_y)   = read_data(args.test)
  print(np.min(train_x), np.max(train_x))
  print(train_x, test_x)
  # Normalize the training data
  (train_x, test_x) = normalize_data(train_x, test_x, 
                          args.rangenorm, args.varnorm, args.exnorm)
  print(train_x, test_x)
  print(np.min(train_x), np.max(train_x))
  acc = runTest(test_x, test_y,train_x, train_y,args.k)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main()
