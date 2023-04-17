from sklearn.metrics import silhouette_score, rand_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from utils import Utils

class Metric:
  def __init__(self, data, true_labels, cluster_labels) -> None:
    self.confusion_matrix = Utils.get_confusion_matrix(true_labels=true_labels, cluster_labels=cluster_labels)

    self.TP = self.confusion_matrix[1][1]
    self.FP = self.confusion_matrix[1][0]
    self.TN = self.confusion_matrix[0][0]
    self.FN = self.confusion_matrix[0][1]

    self.true_labels = true_labels
    self.cluster_labels = cluster_labels
    self.n = len(true_labels)
    self.data = data
    self.contingency_matrix = self.compute_contingency_matrix().transpose()
    
  # Contingency matrix based measures
  def compute_contingency_matrix(self):
    return contingency_matrix(self.true_labels, self.cluster_labels)
  
  # Contingency matrix based measures
  def compute_recall(self):
    total = 0 
    nr_clusters = len(self.contingency_matrix)
    for cluster in self.contingency_matrix: 
      idx = cluster.argmax()
      n_ij = max(self.contingency_matrix[:, idx])
      m_j = sum(self.contingency_matrix[:, idx])
      total = total + (n_ij / m_j)
    return total / nr_clusters
  
  # Contingency matrix based measures
  def compute_precision(self):
    total = 0 
    nr_clusters = len(self.contingency_matrix)
    for cluster in self.contingency_matrix:
      n_i = sum(cluster) 
      n_ij = max(cluster)
      total = total + (n_ij / n_i)
    return total / nr_clusters
    
  
  # Contingency matrix based measures
  def compute_F_measure(self, beta=1):
    total_f = 0 
    n = len(self.contingency_matrix)
    for cluster in self.contingency_matrix: 
      prec = max(cluster) / sum(cluster) 
      idx = cluster.argmax()
      recall = max(self.contingency_matrix[:, idx]) / sum(self.contingency_matrix[:, idx])
      total_f = total_f + ((2 * recall * prec) / (recall + prec))

    # return ((beta ** 2 + 1) * R * P) / (P * beta ** 2 + R)
    return total_f / n
  
  # Internal measure 
  def compute_silhouette(self):
    return silhouette_score(self.data, self.cluster_labels)
  
  # Pairwise measure 
  def compute_rand_index(self):
    return rand_score(self.true_labels, self.cluster_labels)
  
  # External entropy measure 
  def compute_v_measure(self):
    self.h, self.c, self.v = homogeneity_completeness_v_measure(self.true_labels, self.cluster_labels)
    return self.h, self.c, self.v
  
  def compute_all(self):
    self.compute_v_measure()
    return {"Rand": self.compute_rand_index(), "Silhouette": self.compute_silhouette(), 'Homogeneity': self.h, 'Completeness': self.c, 'V-measure': self.v, 'Recall': self.compute_recall(), 'Precision': self.compute_precision(), 'F-measure': self.compute_F_measure(beta=1) }
    
  # def compute_calinski_harabasz_score(self):
  #   return calinski_harabasz_score(self.data, self.cluster_labels)
  
  # def compute_davies_bouldin_score(self):
  #   return davies_bouldin_score(self.data, self.cluster_labels)

  # def compute_accuracy(self):
  #   '''
  #     This is the same as Rand index in terms of clustering. 
  #   '''
  #   return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)