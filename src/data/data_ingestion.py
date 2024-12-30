import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

logger=logging.getLogger('Data Ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

formattor=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formattor)
file_handler.setFormatter(formattor)

logger.addHandler(console_handler)
logger.addHandler(console_handler)


def load_params(params_path:str)->dict:
    # Load params form YAML file
    try:
        with open(params_path,'r')as file:
            params=yaml.safe_load(file)
        logger.debug('Parameter receiver from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error %s',e)
        raise
    except Exception as e:
        logger.error('Exception error %s',e)
        raise

def load_data(data_url:str)->pd.DataFrame:
    # Load data from CSV file
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to Parse the CSV file %e',e)
        raise
    except Exception as e:
        logger.error('Unexcepted occured while loading the data %s',e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()