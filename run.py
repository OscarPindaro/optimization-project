import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.preprocessing import MinMaxScaler


class Run():

    def __init__(self, dataset_path, load_iris=False):
        if load_iris:
            X, y = datasets.load_iris(as_frame=True, return_X_y=True)
            iris = pd.DataFrame(X)
            iris["Classes"] = y
            self.df = iris
        else:
            raise NotImplementedError()
        self.scaler = MinMaxScaler()
        self.column_names = list(self.df)
        self.index_features = list(range(0, len(self.df.columns) - 1))
        self.classes = self.df["Classes"].unique().toList()
        self.classes_en = [i for i in range(len(self.classes))]
        self.class_encoder = preprocessing.LabelEncoder()

    def prepare_dataset(self):
        df_std = self.df.copy()
        self.class_encoder.fit(df_std["Classes"])
        df_std["Classes"] = self.class_encoder.transform(df_std['Classes'])
        df_std[self.column_names[0:-1]] = self.scaler.fit_transform(df_std[self.column_names[0:-1]])


