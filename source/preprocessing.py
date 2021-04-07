import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path


############################################
# Global Variables
############################################
DATA_PATH = '../data/trip/'
Drivers = [
    'driver_A', 'driver_B', 'driver_F',
    'driver_G', 'driver_I'
]
TRIP_CNT = 10
TEST_TRIP = 9


############################################
# Interface Functions of preprocessing
############################################


def get_dataset(filename) -> pd.DataFrame:
    """csv file을 읽어서 DataFrame 형태로 반환한다.

    Args:
        filename: target file

    Returns:
        DataFame 형태의 주행 dataset
    """
    print('[#] Get dataset ...', filename)
    df = pd.read_csv(filename).rename(columns=lambda x: x.strip())

    return df.reset_index(drop=True)


def get_driver_dataset(driver='A', trip=0) -> pd.DataFrame:
    """특정 Driver의 Trip 주행 dataset을 가져온다.

    Args:
        driver: A, B, G, F, I
        trip: 0 ~ 9

    Returns:
        DataFame 형태의 주행 dataset
    """
    print('[#] Get driver dataset ...')
    trip = 0 if trip < 0 else 9 if trip > 9 else trip

    dict = {'A': 0, 'B': 1, 'F': 2, 'G': 3, 'I': 4}
    driver = Drivers[dict[driver]]
    target = _get_file_name(driver, trip)
    print('[*] filename:', target)
    df = pd.read_csv(target).rename(columns=lambda x: x.strip())

    return df.reset_index(drop=True)


def get_test_dataset(short_term=0, save=None) -> pd.DataFrame:
    """각 Driver의 마지막 (9번) Trip data를 모아서 Test dataset을 구성한다.

        Args:
            short_term: 0 ~ 30 -> 0이면, 9번 trip data의 전체 data를 반환하고,
            그렇지 않은 경우 minutes 단위의 누적 data를 반환한다.
            save: None (default) or filename -> dataset을 csv 파일로 저장한다.

        Returns:
            DataFrame 형태의 test dataset
        """
    print('[#] Get test dataset ...')
    short_term = 0 if short_term < 0 else 30 if short_term > 30 else short_term

    df_list = list()
    for driver in Drivers:
        target = _get_file_name(driver, TEST_TRIP)
        print('[*] insert file:', target)
        tmp = pd.read_csv(target).rename(columns=lambda x: x.strip())
        if short_term == 0:
            df_list.append(tmp)
        else:
            df_list.append(tmp[:short_term*60])

    df = pd.concat(df_list)
    del df_list

    if save is not None:
        print('[*] save file:', save)
        df.to_csv(save, index=False)

    return df.reset_index(drop=True)


def get_train_dataset(trip=8, save=None) -> pd.DataFrame:
    """각 Driver의 trip data까지의 data를 모아서 Train dataset을 구성한다.

        Args:
            trip: 0 ~ 8
            save: None (default) or filename -> dataset을 csv 파일로 저장한다.

        Returns:
            DataFrame 형태의 train dataset
        """
    print('[#] Get train dataset ...')
    trip = 0 if trip < 0 else 8 if trip > 8 else trip

    df_list = list()
    for driver in Drivers:
        for cnt in range(trip):
            target = _get_file_name(driver, cnt)
            print('[*] insert file:', target)
            df_list.append(pd.read_csv(target).rename(columns=lambda x: x.strip()))

    df = pd.concat(df_list)
    del df_list

    if save is not None:
        print('[*] save file:', save)
        df.to_csv(save, index=False)

    return df.reset_index(drop=True)


def merge_dataset(*args) -> pd.DataFrame:
    """가변 인자로 받은 DataFrame 형태의 dataset을 merge하여 새로운 dataset을 구성한다.

        Args:
            args: datasets

        Returns:
            DataFrame 형태의 merged dataset.
        """
    print('[#] Merge dataset ...')

    df_list = list()
    for data in args:
        df_list.append(data)
        del data
    df = pd.concat(df_list)
    del df_list

    return df.reset_index(drop=True)


def hdf5_to_csv(save=True) -> None:
    """drirver 및 trip별로 구분된 *.h5 파일들을 csv file로 변경하여 저장한다.

        Args:
            save: file save option -> True (default) / False

        Returns:
            None
        """
    print('[#] Merge dataset ...')
    drivers = ['driver_A', 'driver_B', 'driver_F', 'driver_G', 'driver_I']

    for driver in drivers:
        driver = DATA_PATH + driver
        targets = sorted([x for x in Path(driver).glob("*.h5")])
        if save is False:
            print(targets)
            continue
        for i, _ in enumerate(targets):
            _save_csv_from_hdf5(driver, trip_cnt=i)
        _save_csv_from_hdf5(driver, trip_cnt=None)


def load_data(tr_file, te_file):
    """각 Driver의 trip data까지의 data를 모아서 Train dataset을 구성한다.

        Args:
            trip: 0 ~ 8
            save: None (default) or filename -> dataset을 csv 파일로 저장한다.

        Returns:
            train_data, train_label, test_data, test_label
        """
    print('[#] Loading data:', tr_file, te_file)
    tr_df = get_dataset(tr_file)
    te_df = get_dataset(te_file)

    tr_X = tr_df.iloc[:, 0:1092];
    tr_y = tr_df['Driver']
    te_X = te_df.iloc[:, 0:1092];
    te_y = te_df['Driver']

    del tr_df; del te_df
    return tr_X, te_X, tr_y, te_y


############################################
# Local Functions of preprocessing
############################################


def _get_file_name(driver, trip) -> str:
    return DATA_PATH + driver + '/' + driver + '_' + str(trip) + '.csv'


def _make_data_cols_name():
    cols = []
    for i in range(1092):
        cols.append('F-' + str(i+1))
    return cols


def _save_csv_from_hdf5(path, trip_cnt=None):
    if trip_cnt is None:
        file_name = path + '/' + path.split('/')[-1] + '_all'
    else:
        file_name = path + '/' + path.split('/')[-1] + '_' + str(trip_cnt)

    print("[*] Data Loading from hdf5: ", file_name)
    def _read_from_hdf5(target):
        with h5py.File(target, "r") as f:
            a_group_key = list(f.keys())[0]
            return list(f[a_group_key])
    targets = sorted([x for x in Path(path).glob("*.h5")])

    label = path.split('/')[-1].split('_')[-1]
    if trip_cnt is None:
        lists = list()
        for target in tqdm(targets, total=len(targets)):
            lists.extend(_read_from_hdf5(target))
        print("[*] Transform h5 to csv ...")
        df = pd.DataFrame(lists, columns=_make_data_cols_name())
    else:
        print("[*] Transform h5 to csv ...")
        df = pd.DataFrame(_read_from_hdf5(targets[trip_cnt]), columns=_make_data_cols_name())

    df['Driver'] = label
    df = df.reset_index(drop=True)

    file_name = file_name + '.csv'
    print('[*] Save csv file: ', file_name)
    df.to_csv(file_name, index=False)


if __name__=='__main__':
    None