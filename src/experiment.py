from clustering import Clustering
from constants import constants
from figures import Figures
from metric import Metric
from utils import Utils
import pickle

class Experiment: 
  def run_experiment_base(is_interval=True):
    metrics = {}
    cluster_distributions = {}

    # KB BERT 
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.KB_bert_finetuned, constants.KB_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB bert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'KB bert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB bert'] = true_label_distribution

    # KB ALBERT 
    data, true_labels, cluster_labels = Clustering.cluster_albert(constants.KB_albert_finetuned, constants.KB_albert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB albert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'KB albert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB albert'] = true_label_distribution

    # KB SBERT
    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.KB_sbert_finetuned, constants.KB_sbert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB sbert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'KB sbert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB sbert'] = true_label_distribution

    # AF BERT 
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert_finetuned, constants.AF_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['AF bert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'AF bert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['AF bert'] = true_label_distribution

    # ML BERT 
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert_finetuned, constants.ML_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML bert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'ML bert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['ML bert'] = true_label_distribution

    # ML SBERT
    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.ML_sbert_finetuned, constants.ML_sbert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML sbert'] = M.compute_all()
    cm = M.compute_contingency_matrix()
    Figures.contingency_matrix_figure(cm, 'ML sbert')

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB sbert'] = true_label_distribution
    
    print(metrics)
    file = open('eval_metrics_base.pkl', 'wb')
    pickle.dump(metrics, file)
    file.close()

    print(cluster_distributions)
    file = open('label_distribution_base.pkl', 'wb')
    pickle.dump(cluster_distributions, file)
    file.close()

  def run_experiment_improve():
    is_interval = False
    metrics = {}
    cluster_distributions = {}


    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.KB_bert_finetuned, constants.KB_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB bert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB bert'] = true_label_distribution

    data, true_labels, cluster_labels = Clustering.cluster_albert(constants.KB_albert_finetuned, constants.KB_albert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB albert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB albert'] = true_label_distribution

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.KB_sbert_finetuned, constants.KB_sbert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB sbert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['KB sbert'] = true_label_distribution

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert_finetuned, constants.AF_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['AF bert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['AF bert'] = true_label_distribution

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert_finetuned, constants.ML_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML bert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['ML bert'] = true_label_distribution

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.ML_sbert_finetuned, constants.ML_sbert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML sbert'] = M.compute_all()

    true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, cluster_labels)
    cluster_distributions['ML sbert'] = true_label_distribution
    
    print(metrics)
    file = open('eval_metrics_improv.pkl', 'wb')
    pickle.dump(metrics, file)
    file.close()

    print(cluster_distributions)
    file = open('label_distribution_improv.pkl', 'wb')
    pickle.dump(cluster_distributions, file)
    file.close()