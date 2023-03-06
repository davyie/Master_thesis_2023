# def get_indices(self, cluster_id, arr):
  #    return list(filter(lambda i: arr[i] == cluster_id, range(len(arr))))
  
  # def get_labels_with_indices(self, indices, true_labels):
  #    return [true_labels[i] for i in indices] # This returns a list of labels [neg, very neg,....]
  
  # def get_counts(self, labels):
  #    return Counter(labels)
  
  def get_cluster_labels(true_labels, pred_labels, n_clusters):
    for cluster_id in range(n_clusters): 
      indices = list(filter(lambda i: pred_labels[i] == cluster_id, range(len(pred_labels))))
      label_ids = [true_labels[i] for i in indices]
      count = Counter(label_ids)
      cluster_label = count.most_common()[0][0]
      print('cluster: {}, most_common: {}'.format(cluster_id, cluster_label), {k: count[k] for k in sorted(count)})