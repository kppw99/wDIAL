import neptune

import pickle

from helpers import *
from source.models import *

############################################
# DATASET CONFIGURATIONS
############################################


do_record = False # Neptune 이라는 ML Logging tool을 이용할 때만 True로 사용합니다

train_path = '../data/train_dataset/'
test_path = '../data/test_dataset/'

print('Train: ', len(os.listdir(train_path)))
print('Test: ', len(os.listdir(test_path)))

print(os.listdir(test_path))

# driver_dict = {
#     'B': 0,
#     'F': 1,
#     'G': 2,
#     'I': 3
# }

# init_train_{}, init_train_{}_label: train_x, train_y 입니다
# pool_{}: inititial train set에 들어가지 않은 데이터들이 dict 형태로 들어가있습니다
init_train_B, init_train_B_label, pool_B = create_driver_arrays(train_path, 'B', 0)
init_train_F, init_train_F_label, pool_F = create_driver_arrays(train_path, 'F', 0)
init_train_G, init_train_G_label, pool_G = create_driver_arrays(train_path, 'G', 0)
init_train_I, init_train_I_label, pool_I = create_driver_arrays(train_path, 'I', 0)

test_x, test_y = create_test_set(test_path)

############################################
# ITERATING LOOP
############################################

acc_list = list()
prec_list = list()
rec_list = list()
f1_list = list()
train_set_config_list = list()

pool_cnt = 4 # 한 번에 몇 datapoint씩 파싱할 것인지 선택
train_set_configs = {
    'B': 1,
    'F': 1,
    'G': 1,
    'I': 1
}

pool_keys = list(pool_B.keys()) + list(pool_F.keys()) + list(pool_G.keys()) + list(pool_I.keys())
pool_keys = np.asarray(pool_keys)

pool_merged = {**pool_B, **pool_F, **pool_G, **pool_I}

print(len(pool_keys))
print(len(pool_merged))

for mode in ['distance', 'random']:   # 'random' and 'distance'
    # pool에 남아있는 개수 / 한 번에 몇개씩 pool에서 뽑을지?
    for iteration_idx in range(int(len(pool_keys)/pool_cnt)):
        print('-------------------------------------------------')
        print('[#] Phase ', iteration_idx)
        print('-------------------------------------------------')

        # 처음에는 그냥 모델을 학습시키고 끝냄 -> 바로 oracle에 넣어줌
        if iteration_idx == 0:
            # init_data 들로 train set을 구성하고
            train_x = np.concatenate((init_train_B, init_train_F, init_train_G, init_train_I), axis=0)
            train_y = np.concatenate((init_train_B_label, init_train_F_label, init_train_G_label, init_train_I_label), axis=0)

            if do_record:
                neptune.init(
                    # 'accida/sandbox',
                    # api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOWNlZmJkMDYtODI2Ny00NWM5LTkwZmQtYjUxMDFmM2FlYWU0In0=')
                    'kppw99/wDIAL',
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NmMxZWNlNC1hYzIxLTRlMmYtODlmZi0zZjhkNzBiNjE3M2UifQ==')
            
                # 실험 생성
                experiment_object = neptune.create_experiment(
                    name='accida_binary_kp',
                )
                neptune.append_tag('wDIAL')
                neptune.append_tag('mode: ' + mode)
                neptune.append_tag('distance: Minimum')

        else:
            if mode == 'random':
                print('[#] Start Random Values from the Pool..')
                np.random.seed(10)
                np.random.shuffle(pool_keys)
                selected_keys = pool_keys[:pool_cnt]
            else:
                print('[#] Start Querying to the Oracle..')
                # Oracle에 Query 해서 SVM 거리들과의 거리를 체크
                pool_dist_dict = dict()

                print(' -> Start Calculating Distances..')
                for pool_key in pool_keys:
                    pool_selected_data = pool_merged[pool_key]
                    pool_distance = calculate_distance(model, pool_selected_data)
                    pool_dist_dict[pool_key] = pool_distance

        #             print(pool_key, ' : ', pool_distance)

                # pool_cnt 개수만큼 데이터를 추출해둠 (거리가 먼 것 부터 선택)
                selected_pool = sorted(pool_dist_dict.items(), key=lambda x: x[1], reverse=True)[:pool_cnt]
                selected_keys = [i[0] for i in selected_pool]

            # Training Set에 총 몇 개의 데이터가 들어갔는지 확인
            print('selected_key: ', selected_keys)
            for selected_key in selected_keys:
                train_set_configs[selected_key[0]] = train_set_configs[selected_key[0]] + 1

            # 선택된 데이터를 추가해서 새로운 train set을 꾸림
            train_x, train_y = build_added_train_set(pool_merged, train_x, train_y, selected_keys)

            # 이미 선택된 key 값들은 삭제시킴
            pool_keys = np.setdiff1d(pool_keys, selected_keys)

            # 이미 선택된 Data 들은 삭제시킴
            [pool_merged.pop(selected_key) for selected_key in selected_keys]

        model, acc, prec, rec, f1 = svm(train_x, train_y, test_x, test_y)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        train_set_config_list.append(train_set_configs)

        print('train_x: ', train_x.shape)
        print(train_set_configs)
        print('Number of the Pool: ', len(pool_keys))
        
        if do_record:
            neptune.log_metric('Accuracy', acc)
            neptune.log_metric('Precision of Class 0: ', prec[0])
            neptune.log_metric('Precision of Class 1: ', prec[1])
            neptune.log_metric('Precision of Class 2: ', prec[2])
            neptune.log_metric('Precision of Class 3: ', prec[3])
            
            neptune.log_metric('Recall of Class 0: ', rec[0])
            neptune.log_metric('Recall of Class 1: ', rec[1])
            neptune.log_metric('Recall of Class 2: ', rec[2])
            neptune.log_metric('Recall of Class 3: ', rec[3])
            
            neptune.log_metric('F1 Score of Class 0: ', f1[0])
            neptune.log_metric('F1 Score of Class 1: ', f1[1])
            neptune.log_metric('F1 Score of Class 2: ', f1[2])
            neptune.log_metric('F1 Score of Class 3: ', f1[3])

    log_dict = {
        'acc_list': acc_list,
        'prec_list': prec_list,
        'rec_list': rec_list,
        'f1_list': f1_list,
        'train_set_config_list': train_set_config_list
    }

    with open('./exp_active_' + mode + '.pkl', 'wb') as f:
        pickle.dump(log_dict, f)