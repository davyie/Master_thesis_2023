from misc import print_results

def main():
  # kMeans_Experiment().run_base_kMeans(constants.absa_labels_eng.values(), [i for i in range(0,5)], constants.absa_labels_eng.keys(), constants.embeddings_absa, constants.embedding_folder_absa, 5, 'true_labels.pkl' , 'result_absa/')
  # kMeans_Experiment().run_improv_kMeans(constants.absa_labels_eng.values(), [i for i in range(0,5)], constants.absa_labels_eng.keys(), constants.embeddings_absa, constants.embedding_folder_absa, 5, 'true_labels.pkl' , 'result_absa/')

  # kMeans_Experiment().run_base_kMeans(constants.overlim_labels_eng.values(), [i for i in range(0,2)], constants.overlim_labels_eng.keys(), constants.embeddings_sst, constants.embedding_folder_sst, 2, 'true_labels.pkl' , 'result_sst/')
  # kMeans_Experiment().run_improv_kMeans(constants.overlim_labels_eng.values(), [i for i in range(0,2)], constants.overlim_labels_eng.keys(), constants.embeddings_sst, constants.embedding_folder_sst, 2, 'true_labels.pkl' , 'result_sst/')
  
  # OverLim().print_data_distribution()

  print_results()

  pass


if __name__ == '__main__': 
  main()

