import pandas as pd

from constants import constants 

class DataLoader: 
    def __init__(self, path) -> None:
        '''
          This is the constructor of the class. 
          @path - this is the path from this position to the dataset 
        '''
        self.data = self.load(path)
        return 

    def get_data_by_col_name(self, col_name):
        '''
          This method returns the given column from the data. 
        '''
        return self.data[col_name]
    
    def get_all_data(self):
        '''
          Return all data 
        '''
        return self.data
    
    def get_data_with_labels(self):
        '''
          Return a list of tuple with text and sentiment words. 
        '''
        
        self.data.average = map(round,  self.data.average)

        def replace_score_with_label(score): # Local method
            return constants.labels[score]
        # Replace all integers with words. 
        self.data.average = map(replace_score_with_label, self.data.average)

        return list(zip(self.data.text, self.data.average))
    
    def get_labels(self):
        '''
          Return labels
        '''
        return self.labels

    def load(self, path):
      '''
        This method is used to load in the TSV data 
      '''
      return pd.read_csv(path, sep='\t')