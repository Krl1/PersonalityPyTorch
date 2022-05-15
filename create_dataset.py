import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from params import RANDOM_SEED, LocationConfig, CreateDataConfig


def create_new_data_directories():
    Path(LocationConfig.new_data).mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.new_data + 'train').mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.new_data + 'test').mkdir(exist_ok=True, parents=True)
    
    
def get_short_video_name(videoNames):
    ShortVideoName = []
    for videoName in videoNames.values:
        ShortVideoName.append(videoName.split('.')[0])
    return ShortVideoName

def create_mean_video_name_df(df):
    cols = ['ValueExtraversion','ValueAgreeableness','ValueConscientiousness','ValueNeurotisicm','ValueOpenness','ShortVideoName']
    grouped_df = df[cols].groupby('ShortVideoName')
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()
    return mean_df
    

def create_conntected_dataset():
    df = pd.read_csv(LocationConfig.raw_data + 'bigfive_labels.csv')

    df['ShortVideoName'] = get_short_video_name(df['VideoName'])

    mean_df = create_mean_video_name_df(df)
    mean_df.to_csv(LocationConfig.raw_data + 'bigfive_labels_mean.csv')
    mean_df = mean_df.set_index('ShortVideoName')
    
    X_train, X_test = train_test_split(
        np.array(mean_df.index), 
        test_size=CreateDataConfig.test_size_ratio,
        random_state=RANDOM_SEED
    )
    images_dict_train = {'X':[], 'Y':[]}
    images_dict_test = {'X':[], 'Y':[]}
    for image_path in tqdm(Path(LocationConfig.raw_data).glob('*/*.jpg'), total=30935):
        X = cv2.imread(str(image_path))    
        image_group = image_path.name.split('.')[0]
        image_no = image_path.name.split('.')[2][-5:]
        Y = mean_df.loc[image_group].values
        if CreateDataConfig.classification:
            Y = list(np.where(Y>CreateDataConfig.Y_threshold, 1, 0))
            
        if image_group in X_test:
            images_dict_test['X'].append(X)
            images_dict_test['Y'].append(Y)
        else:
            images_dict_train['X'].append(X)
            images_dict_train['Y'].append(Y)
            
    with open(LocationConfig.new_data + 'train/train.pickle', 'wb') as handle:
        pickle.dump(images_dict_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(LocationConfig.new_data + 'test/test.pickle', 'wb') as handle:
        pickle.dump(images_dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_new_data_directories()
    if CreateDataConfig.connect:
        create_conntected_dataset()