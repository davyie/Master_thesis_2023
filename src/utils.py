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