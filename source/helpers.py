import os
import numpy as np
import pandas as pd


driver_dict = {
    'B': 0,
    'F': 1,
    'G': 2,
    'I': 3
}


# train_path, driver 이름 (B, F, G..), index를 넣으면 해당하는 순서의 csv파일들을 읽어옵니다
def create_driver_arrays(path, driver_name, selected_idx):
    print('Working on ', driver_name, ' ...')
    
    filenames = sorted([i for i in os.listdir(path) if driver_name in i])
    
    pool = dict()
    
    for file_idx in range(len(filenames)):
        filepath = os.path.join(path, filenames[file_idx])
        data = pd.read_csv(filepath)
        data = data.drop(['Driver'], axis=1).values
        
        # 선택된 idx는 처음 모델을 학습시킬 떄 initial dataset으로 사용하고
        if file_idx == selected_idx:
            init_data = data
        
        # 나머지 idx의 데이터들은 unlabeled pool로 사용합니다
        else:
            pool[driver_name + '_' + str(file_idx)] = data
            
    init_label = [driver_dict[driver_name] for i in range(len(init_data))]
            
    return init_data, init_label, pool


# test set directory path를 넣어주면, test_x, test_label을 생성 후 return 해줍니다
def create_test_set(path):
    filenames = os.listdir(path)
    
    idx = 0
    for key, values in driver_dict.items():
        
        filename = [i for i in filenames if key in i][0]
        filepath = os.path.join(path, filename)
        data = pd.read_csv(filepath)
        
        data = data.drop(['Driver'], axis=1).values
        label = np.asarray([driver_dict[key] for i in range(len(data))])
        
        if idx == 0:
            init_data = data
            init_label = label
            
        else:
            init_data = np.concatenate((init_data, data), axis=0)
            init_label = np.concatenate((init_label, label), axis=0)
            
        idx += 1

    return init_data, init_label
    

# 기존에 있던 train_x, train_y를 parameter로 넣어주면 업데이트된 train_x, train_y를 return 해줍니다
def build_added_train_set(pool_merged, train_x, train_y, pool_keys):
    for pool_key in pool_keys:
        # print('>> pool_key: ', pool_key)
        
        data = pool_merged[pool_key]
        label = np.asarray([driver_dict[pool_key[0]] for i in range(len(data))])
        
        # print(data[:10])
        # print(label[:10])
        
        train_x = np.concatenate((train_x, data), axis=0)
        train_y = np.concatenate((train_y, label), axis=0)
    
    return train_x, train_y