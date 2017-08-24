import pickle
import logging

import pandas as pd


logger = logging.getLogger(__name__)


class Pickle(object):
    def __init__(self, path, **kwargs):
        self._path = path

    def load(self):
        with open(self._path, 'rb') as f:
            df = pickle.load(f)
        return df

    def save(self, df):
        with open(self._path, 'wb') as f:
            pickle.dump(df, f)


class CSV(object):
    def __init__(self, path, **kwargs):
        self._path = path
        self._kwargs = kwargs

    def load(self):
        return pd.read_csv(self._path, **self._kwargs)

    def save(self, df):
        if df.index.name is not None:
            logger.warning(
                "You are saving dataframe with non-trivial index. "
                "It will be lost. Reset it or consider using PickleTarget."
            )
        df.to_csv(self._path, index=False, **self._kwargs)


class Shapefile(object):
    def __init__(self, path, **kwargs):
        self._path = path
        self._kwargs = kwargs

    def load(self):
        return gpd.read_file(self._path, **self._kwargs)

    def save(self, df):
        if df.index.name is not None:
            logger.warning(
                "You are saving dataframe with non-trivial index. "
                "It will be lost. Reset it or consider using PickleTarget."
            )
        df.to_file(self._path, **self._kwargs)


class Bcolz(object):
    pass

