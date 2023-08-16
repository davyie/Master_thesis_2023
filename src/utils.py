from collections import Counter
from itertools import combinations, product
import numpy as np 
import pickle

class Utils:
    
  def from_tensor_to_nparray(tensor):
      '''
      This method helps us convert a tensor to a np array which is used in 
      skleanr K-means algorithm. 
      Use this before running K-means.
      @param tensor - a tensor object 
      '''
      return tensor.detach().numpy()
  
  def get_cls_from_batch(data):
     '''
     This method extracts the CLS token embedding from a batch of processed sequences.
     The dimension of data is [batch, #token, hidden_size].
     The solution is to keep the batch and hidden size but we want the first. 
     Therefore it is a 0 at the middle. 
     Use this when obtained BERT embeddings. 
     @param data - This argument has the type tensor 
     '''
     return data.last_hidden_state[:, 0, :]
  
  def get_str_list_of_labels(label_dict):
     '''
     This method returns a str list of labels. 
     @param label_dict - this is the label dictionary from constants.
     '''
     return list(label_dict.values())
  
  def from_series_to_list(series):
     return series.tolist()

  def get_true_labels_per_cluster(true_labels, cluster_ids, nr_clusters=5):
    '''
    This function is used to draw figures. It returns a dictionary of the 
    format {cluster_id: Counter({true_label: count})} which is sent to 
    draw method. Use this function after we have run embedding and K-means. 
    It takes two arguements, true_labels and cluster_ids. 
    @param true_labels - This is a list of true labels 
    @param cluster_ids - This is a list of cluster ids 
    '''
    cluster_label_dict = {}
    for cluster_id in range(nr_clusters):
      indices = [idx for idx in range(len(cluster_ids)) if cluster_ids[idx] == cluster_id ]
      true_labels_cluster = [true_labels[i] for i in indices]
      counts = Counter(true_labels_cluster)
      for i in range(1, 6): # True label start from 1 up to 5 
        if i not in counts.keys():
          counts[i] = 0
      cluster_label_dict[cluster_id] = counts
    return cluster_label_dict # {0: [1, 2, 1, 1, 2, 3...], 1: [1, 3, 3...], ...}

  def get_confusion_matrix(true_labels, cluster_labels):
    '''
      This method computes confusion matrix. The computation is done for 
      each PAIR in the dataset. 
      The out is as follows: 
      [ 
        [<True neg>, <False Neg>],
        [<False Pos, <True Pos>] 
      ]
      True negative is defined as when true labels DISAGREE and algorithm label DISAGREE. 
      True positive is defined as wehn true labels AGREE and algorithm label AGREE. 
      False negative is defined as when true labels AGREE and algorithm label DISAGREE.
      False positive is defined as when true labels DISAGREE and algorithm label AGREE. 
    '''
    m = len(cluster_labels)
    n = len(true_labels)
    if n != m: # Error message 
      print("Different lengths! C_labels: {} | T_labels: {}".format(m, n))
      return 
    confusion_matrix = [[0 for _ in range(2)] for _ in range(2)]
    m = len(cluster_labels)
    n = len(true_labels)
    pairwise_points = list(combinations(range(n - 1), 2))
    for (a, b) in pairwise_points: 
      c_a_label = cluster_labels[a]
      t_a_label = true_labels[a]
      c_b_label = cluster_labels[b]
      t_b_label = true_labels[b] # cluster labels start with 1 
      if c_a_label == c_b_label and t_a_label == t_b_label: # True positive. Agree on cluster and label
        confusion_matrix[1][1] += 1
      elif c_a_label == c_b_label and t_a_label != t_b_label: # False positive 
        confusion_matrix[1][0] += 1
      elif c_a_label != c_b_label and t_a_label == t_b_label: # False negative. 
        confusion_matrix[0][1] += 1
      else: # True negative 
        confusion_matrix[0][0] += 1 
    return confusion_matrix 
        

  def print_to_file(filename, metrics):
    with open(filename, 'w') as out:
      for (name, metric) in metrics.items():
        out.write( "{}: {}\n".format(name, metric))

  def get_intervals(data, batch_size=64):
    intervals = []
    current = 0
    while True: 
      if current + batch_size > len(data):
        intervals.append(slice(current, len(data)))
        break
      intervals.append(slice(current, current + batch_size))
      current += batch_size
    return intervals

  def read_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data