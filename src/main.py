from constants import constants

from data_loader import DataLoader

def main():
  DL = DataLoader(constants.absa_file_path)

  data = DL.get_data_with_labels()
  print(data)

if __name__ == '__main__': 
  main()
