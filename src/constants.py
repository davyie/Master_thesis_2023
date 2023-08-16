class constants:
  '''
    This class contains constants for the project 
  '''
  absa_labels = {1: 'mycket negativt', 2: 'negativt', 3: 'neutral', 4: 'positivt', 5: 'mycket positivt' }
  overlim_sst_labels = {0: 'negativt', 1: 'positivt'}
  absa_labels_eng = {1: 'very negative', 2:'negative', 3: 'neutral', 4: 'positive', 5: 'very positive'}

  absa_file_path = '../dataset/absabank_imm/P_annotation.tsv'
  overlim_train_file_path = '../dataset/OverLim-sst-sv/sst/train.jsonl'
  overlim_val_file_path = '../dataset/OverLim-sst-sv/sst/val.jsonl'
  overlim_test_file_path = '../dataset/OverLim-sst-sv/sst/test.jsonl'
  overlim_labels_eng = {1: 'negative', 2: 'positive'}

  # Model names 
  KB_bert = 'KBLab/bert-base-swedish-cased'
  KB_sbert = 'KBLab/sentence-bert-swedish-cased'
  KB_albert = 'KBLab/albert-base-swedish-cased-alpha'
  AF_bert = 'af-ai-center/bert-base-swedish-uncased'
  ML_bert = 'bert-base-multilingual-cased'
  ML_sbert = 'sentence-transformers/distiluse-base-multilingual-cased-v2'


  KB_bert_finetuned = './models/KBLab/bert-base-swedish-cased_finetuned'
  KB_albert_finetuned = './models/KBLab/albert-base-swedish-cased-alpha_finetuned'
  KB_sbert_finetuned = './models/KBLab/sentence-bert-swedish-cased_finetuned'
  AF_bert_finetuned = './models/af-ai-center/bert-base-swedish-uncased_finetuned'
  ML_bert_finetuned = './models/bert-base-multilingual-cased_finetuned'
  ML_sbert_finetuned = './models/sentence-transformers/distiluse-base-multilingual-cased-v2_finetuned'

  KB_bert_finetuned_sst = './models-sst/KBLab/bert-base-swedish-cased_finetuned'
  KB_albert_finetuned_sst = './models-sst/KBLab/albert-base-swedish-cased-alpha_finetuned'
  KB_sbert_finetuned_sst = './models-sst/KBLab/sentence-bert-swedish-cased_finetuned'
  AF_bert_finetuned_sst = './models-sst/af-ai-center/bert-base-swedish-uncased_finetuned'
  ML_bert_finetuned_sst = './models-sst/bert-base-multilingual-cased_finetuned'
  ML_sbert_finetuned_sst = './models-sst/sentence-transformers/distiluse-base-multilingual-cased-v2_finetuned'
  
  model_names = ['KB BERT', 'KB SBERT', 'KB ALBERT', 'AF BERT', 'ML BERT', 'ML SBERT']

  # This is OverLim 
  embedding_folder_sst = 'finetuned_embeddings_sst/'
  embeddings_sst = ['KB BERT_embeddings.pkl', 'KB SBERT_embeddings.pkl', 'KB ALBERT_embeddings.pkl', 'AF BERT_embeddings.pkl', 'ML BERT_embeddings.pkl', 'ML SBERT_embeddings.pkl']
  initial_points_embeddings_sst = ['KB BERT_initial_points_embeddings.pkl', 'KB SBERT_initial_points_embeddings.pkl', 'KB ALBERT_initial_points_embeddings.pkl', 'AF BERT_initial_points_embeddings.pkl', 'ML BERT_initial_points_embeddings.pkl', 'ML SBERT_initial_points_embeddings.pkl']
  # This is ABSABank 
  embedding_folder_absa = 'finetuned_embeddings_absa/'
  embeddings_absa = ['KB BERT_embeddings.pkl', 'KB SBERT_embeddings.pkl', 'KB ALBERT_embeddings.pkl', 'AF BERT_embeddings.pkl', 'ML BERT_embeddings.pkl', 'ML SBERT_embeddings.pkl']
  initial_points_embeddings = ['KB BERT_initial_points_embeddings.pkl', 'KB SBERT_initial_points_embeddings.pkl', 'KB ALBERT_initial_points_embeddings.pkl', 'AF BERT_initial_points_embeddings.pkl', 'ML BERT_initial_points_embeddings.pkl', 'ML SBERT_initial_points_embeddings.pkl']
  

  