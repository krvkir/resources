import os
import pickle
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def _ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return f


class Pickle(object):
    def __init__(self, path, **kwargs):
        self._path = path

    def load(self):
        with open(self._path, 'rb') as f:
            df = pickle.load(f)
        return df

    def save(self, df):
        with open(_ensure_dir(self._path), 'wb') as f:
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
        df.to_csv(_ensure_dir(self._path), index=False, **self._kwargs)


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
        df.to_file(_ensure_dir(self._path), **self._kwargs)


class Bcolz(object):
    pass


def cache(*resources):
    def decorator(fn):
        def wrapped(*args, **kwargs):
            try:
                results = (r.load() for r in resources)
                if len(results) == 1:
                    results = results[0]
                logger.info("Loaded all from cache.")
            except:
                logger.info("Cannot load from cache, evaluating.")
                results = fn(*args, **kwargs)
                if type(results) not in [tuple, list]:
                    results = (results, )
                for result, r in zip(results, resources):
                    r.save(result)
            return results
        return wrapped
    return decorator
