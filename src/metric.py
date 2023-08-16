from sklearn.metrics import silhouette_score, rand_score, homogeneity_completeness_v_measure, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from utils import Utils
import numpy as np

class Metric:
  def __init__(self, data, true_labels, cluster_labels) -> None:
    self.confusion_matrix = Utils.get_confusion_matrix(true_labels=true_labels, cluster_labels=cluster_labels)

    self.TP = self.confusion_matrix[1][1]
    self.FP = self.confusion_matrix[1][0]
    self.TN = self.confusion_matrix[0][0]
    self.FN = self.confusion_matrix[0][1]

    self.true_labels = np.array([label - 1 for label in true_labels])
    self.cluster_labels = cluster_labels
    self.n = len(true_labels)
    self.data = data
    self.contingency_matrix = self.compute_contingency_matrix().transpose()
    
    
  # Contingency matrix based measures
  def compute_contingency_matrix(self):
    return contingency_matrix(self.true_labels, self.cluster_labels)
  
  # # Contingency matrix based measures
  def compute_recall(self):
    return self.TP / (self.TP + self.FN)

  def compute_precision(self):
    return self.TP / (self.TP + self.FP)

  def compute_F_measure(self, beta=1):
    return ((beta**2 + 1) * self.compute_precision() * self.compute_recall()) / (beta**2 * (self.compute_precision() + self.compute_recall()))
    
  # Internal measure 
  def compute_silhouette(self):
    return silhouette_score(self.data, self.cluster_labels)
  
  # Pairwise measure 
  def compute_rand_index(self):
    return rand_score(self.true_labels, self.cluster_labels)
  
  # External entropy measure 
  def compute_v_measure(self):
    self.h, self.c, self.v = homogeneity_completeness_v_measure(labels_true=self.true_labels, labels_pred=self.cluster_labels)
    return self.h, self.c, self.v
  
  def compute_h(self):
    h_t_c = 0
    for label in self.contingency_matrix.transpose():
      n_c = sum(label)
      for n_ij in label:
        h_t_c += (n_ij / self.n) * np.log(n_ij / n_c)
    h_t_c = -1 * h_t_c

    h_t = 0 
    for label in self.contingency_matrix.transpose():
      n_c = sum(label)
      h_t += n_c / self.n * np.log(n_c / self.n)
    h_t = -1 * h_t

    return 1 - (h_t_c / h_t)
  
  def compute_c(self):
    h_c_t = 0
    for cluster in self.contingency_matrix:
      n_k = sum(cluster)
      for n_ij in cluster:
        h_c_t += (n_ij / self.n) * np.log(n_ij / n_k)
    h_c_t = -1 * h_c_t

    h_c = 0 
    for cluster in self.contingency_matrix:
      n_k = sum(cluster)
      h_c += n_k / self.n * np.log(n_k / self.n)
    h_c = -1 * h_c

    return 1 - (h_c_t / h_c)

  
  def compute_all(self):
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(self.TP, self.FP, self.TN, self.FN))
    print('Recall: {}, Precision: {}, F1-measure'.format(self.compute_recall(), self.compute_precision(), self.compute_F_measure()))
    self.compute_v_measure()
    return {"Rand": self.compute_rand_index(), "Silhouette": self.compute_silhouette(), 'Homogeneity': self.h, 'Completeness': self.c, 'V-measure': self.v, 'Precision': self.compute_precision(), 'Recall': self.compute_recall(), 'F-measure': self.compute_F_measure()}
    
  # def compute_calinski_harabasz_score(self):
  #   return calinski_harabasz_score(self.data, self.cluster_labels)
  
  # def compute_davies_bouldin_score(self):
  #   return davies_bouldin_score(self.data, self.cluster_labels)

  # def compute_accuracy(self):
  #   '''
  #     This is the same as Rand index in terms of clustering. 
  #   '''
  #   return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)