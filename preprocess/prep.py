import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def combine_multi_file_to_df(filenames):
    assert len(filenames) > 0, "No file to combine"
    data = pd.read_parquet(filenames[0])
    for i in range(1,len(filenames)):
        data = data.append(pd.read_parquet(filenames[i]), ignore_index=True)
    return data

def split_data_with_oversampling(data, test_size=0.2, val_zize=0.2):
    X = pd.DataFrame(data.drop(['Defective'], axis=1))
    y = pd.DataFrame(data['Defective'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

    sm = SMOTE(random_state=1234, k_neighbors=5)  # for oversampling minority data
    X_train, y_train = sm.fit_resample(X_train, y_train)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=val_zize, random_state=1234)

    return X_train, X_test, X_validation, y_train, y_test, y_validation