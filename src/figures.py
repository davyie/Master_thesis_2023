import numpy as np
from constants import constants
from data_loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from hyperparam import result
import pickle

class Figures:
  def data_distribution_figure():
    dl = DataLoader(constants.absa_file_path)
    distribution = dl.get_label_distribution()
    keys = sorted(list(distribution.keys())) # List of keys 
    values = [distribution[k] for k in keys] # list of values 
    
    label_color_swe = {'mycket negativ': '#F0AF73', 'negativ': '#FF675C', 'neutral': '#FFE066', 'positiv': '#C3EB67', 'mycket positiv': '#54FF7D'}
    label_color_eng = {'very negative': '#FF675C', 'negative': '#F0AF73', 'neutral': '#FFE066', 'positive': '#C3EB67', 'very positive': '#54FF7D'}
    title = 'Data distribution'
    filename = 'figures/data_distribution_bar_chart_eng'
    draw_bar(keys, values, y_axis_name="Opinion counts",x_axis_name="Opinion", label_color_dict=label_color_eng, title=title, legend_title="Labels", filename=filename)

  def sample_data_figure():
    dl = DataLoader(constants.absa_file_path)
    data, labels = dl.get_data_with_labels()
    sample_indicies = [labels.index(label) for label in range(1, 6)]
    label_max_len = max(map(len, [constants.absa_labels[i] for i in labels]))
    print('\n')
    print('**********************************************************************')
    print('Label           |  Text')
    print('----------------------------------------------------------------------')
    for i in sample_indicies:
      print(constants.absa_labels[labels[i]].ljust(label_max_len), '| ' , data[i][0:50])
    print('**********************************************************************')
    print('\n')
  
  def true_labels_per_cluster_figure(cluster_labels, model_name):
    '''
    This method takes a dictionary {cluster_id: Counter({true_label: count})}
    '''
    fig, ax = plt.subplots() 
    
    cluster_ids = cluster_labels.keys()
    true_label_count = cluster_labels.values()

    X_axis = np.arange(0.7, len(cluster_ids))
    width = 0.15
    bars = []
    for counter in true_label_count:
      count = counter.values()
      # Draw diagram
      bars.append(ax.bar(X_axis, count, width=width))
      X_axis = X_axis + width

    title = "Number of True labels in each cluster - " + model_name
    filename = "figures/number_of_true_labels_in_cluster_" + model_name

    X_axis = np.arange(0.7, len(cluster_ids))
    labels = [constants.absa_labels[i] for i in range(1, 6)]
    ax.set_title(title)
    ax.set_ylabel(ylabel="True label count")
    ax.set_xlabel(xlabel="True label")
    ax.set_xticks([i for i in range(1,6)])
    ax.set_xticklabels(labels)
    ax.legend(bars, ['Cluster: {}'.format(i + 1) for i in cluster_ids])
    fig.savefig(filename)
    fig.show()
    pass

  def confusion_matrix_figure(conf_matrix):

    conf_matrix = np.array(conf_matrix)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
      for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel(xlabel='Agree true label')
    ax.set_ylabel(ylabel='Agree cluster ID')
    ax.set_title('Confusion Matrix')
    fig.savefig("figures/confusion_matrix")
    fig.show()
  
  def search_hyperparam_figure():
    models = ['KB BERT', 'KB ALBERT', 'KB SBERT', 'AF BERT', 'ML BERT', 'ML SBERT']
    hyperparam_setting = list(result.keys())
    loss = [[item[-1] for item in lst] for lst in list(result.values())]
    # loss = [list(i) for i in zip(*loss)]
    x = np.arange(len(loss))
    print(x)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss)), loss, label=models)
    ax.set_xticks(x)
    ax.set_xticklabels(hyperparam_setting, rotation=30, ha='right')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title("MLM Loss/Hyperparameter search")
    ax.legend()
    plt.ylabel('MLM Loss')
    plt.xlabel('Hyperparameter setting (#epochs, learning rate, batch size)')
    plt.show()
    # print(loss)
    
  def evaluation_metric_figure():
  
    with open('eval_metrics_base.pkl', 'rb') as f:
      metrics = pickle.load(f)
      
    # print(metrics)
    models = metrics.keys()
    mets = {}
    for metric in metrics.values():
      for k, v in metric.items():
        if k not in mets:
          mets[k] = [v]
        else:
          mets[k].append(v)
    
    for k, v in mets.items():
      fig, ax = plt.subplots()
      ax.bar(np.arange(6), mets[k], color=['#F77774', '#B763D4', '#7A85EB', '#63CAD4', '#6CF589', '#F5ED8D'])
      ax.set_xticks(np.arange(6))
      ax.set_xticklabels(models)
      ax.set_xlabel('Model')
      ax.set_ylabel('Metric: ' + k)
      ax.set_title('Evaluation metric ' + k)
    plt.show()

  def contingency_matrix_figure(matrix):
    n = len(matrix)
    m = len(matrix[0])
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap='ocean')

    for i in range(m):
      for j in range(n):
          c = matrix[j, i]
          ax.text(i, j, str(c), va='center', ha='center')

    ax.set_xlabel(xlabel='Cluster ID')
    ax.set_ylabel(ylabel='Partition ID')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    ax.set_title('Contingency Matrix')
    plt.show()