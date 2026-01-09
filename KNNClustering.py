import numpy as np
import statistics

class KNN_Classifier():

  # initiating the parameters
  def __init__(self, distance_metric = 'euclidean'):
    self.distance_metric = distance_metric

  # getting the distance metric
  def get_distance_metric(self, x_train_data_point, x_test_data_point):

    if (self.distance_metric == 'euclidean'):
      distance = 0
      for i in range(len(x_train_data_point) - 1):
        distance += (x_train_data_point[i] - x_test_data_point[i]) ** 2
      return distance ** 0.5

    elif (self.distance_metric == 'manhattan'):
      distance = 0
      for i in range(len(x_train_data_point) - 1):
        distance += abs(x_train_data_point[i] - x_test_data_point[i])
      return distance

    else:
      raise "Invalid distance metric"

  # getting the nearest neighbours
  def nearest_neighbours(self, x_train, x_test_data_point, k):

    distance_list = []
    for x_train_data_point in x_train:
      distance = self.get_distance_metric(x_train_data_point, x_test_data_point)
      distance_list.append((x_train_data_point, distance))

    distance_list.sort(key = lambda x : x[1])

    k_nearest_neighbours = []
    for neighbour in range(k):
      k_nearest_neighbours.append(distance_list[neighbour][0])

    return k_nearest_neighbours

  # predict the class of new data point
  def predict(self, x_train, x_test, k):

    predicted_class = []
    for x_test_data_point in x_test:
      neighbours = self.nearest_neighbours(x_train, x_test_data_point, k)

      label = []
      for data in neighbours:
        label.append(data[-1])

      predicted_class.append(statistics.mode(label))
    return np.array(predicted_class)