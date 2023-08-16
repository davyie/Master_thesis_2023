import pandas as pd
import json 

from constants import constants 
from utils import Utils
from collections import Counter 

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
        
        # def replace_score_with_label(score): # Local method
        #     return constants.absa_labels[score]
        # Replace all integers with words. 
        # self.data.average = map(replace_score_with_label, self.data.average)

        return Utils.from_series_to_list(self.data.text), list(map(round, self.data.average))

    def load(self, path):
      '''
        This method is used to load in the TSV data 
      '''
      return pd.read_csv(path, sep='\t')

    def get_label_distribution(self):
      return Counter(list(map(round, self.data.average)))
    
    def save_to_file(self):
      list_of_sen = Utils.from_series_to_list(self.data.text)
      with open("absa_textdata", 'w') as out:
        for s in list_of_sen:
           out.write(s + '\n')

    def read_json(self, path):
      f = open(path, 'r')
      data = f.readlines()
      labels = [json.loads(r"{}".format(line))['label'] for line in data]
      text_data = [json.loads(r"{}".format(line))['text'] for line in data]
      return text_data, labels
        