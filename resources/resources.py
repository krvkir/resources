import os
import pickle
import logging
from collections import Container
from inspect import signature

import pandas as pd
import bcolz


logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def _ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return f


class Resource(object):
    """
    >>> resource = Resource('/path/to/file')
    >>> resource.path
    '/path/to/file'
    """

    def __init__(self, path):
        self.path = path

    def _get_path(self): return self._path
    def _set_path(self, value): self._path = value
    path = property(_get_path, _set_path, None, "Path to the resource")

    def load(self):
        raise NotImplemented

    def save(self, df):
        raise NotImplemented


class Pickle(Resource):
    def load(self):
        with open(self._path, 'rb') as f:
            df = pickle.load(f)
        return df

    def save(self, df):
        with open(_ensure_dir(self._path), 'wb') as f:
            pickle.dump(df, f)


class CSV(Resource):
    def __init__(self, path, **kwargs):
        super(CSV, self).__init__(path)
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


class Shapefile(Resource):
    def __init__(self, path, **kwargs):
        super(Shapefile, self).__init__(path)
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


class Bcolz(Resource):
    def __init__(self, path, **kwargs):
        super(Bcolz, self).__init__(path)
        self._kwargs = kwargs

    def load(self):
        return bcolz.ctable(rootdir=self._path, **self._kwargs)

    def save(self, df):
        # Check the bcolz restriction on column names.
        assert all(type(s) == str and s.isidentifier() for s in df.columns)
        # Dump data to the disk.
        bcolz.ctable.fromdataframe(
            df, rootdir=_ensure_dir(self._path), **self._kwargs)


def cache(*resources):
    """
    Example 1:
    > cache(Bcolz('data/storage.bcolz'))(my_fn)(arg1, arg2, arg3)

    Example 2:
    > @cache(Bcolz('data/storage.bcolz'))
    > def my_fn(arg1, arg2, arg3):
    >    calc_something_heavy(arg1, arg2, arg3)

    Example 3:
    > @cache(Bcolz, path='data/{date}/storage.bcolz', args=['date', ])
    > def my_func(date, arg1, arg2, arg3):
    >     calc_something_heavy(date, arg1, arg2, arg3)

    """

    def decorator(raw_fn):
        def new_fn(*args, **kwargs):
            # Instantiate all resources which are not instantiated already.
            # Expand path templates with raw function arguments.
            # Prepare arguments.
            argnames_of_raw_fn = tuple(signature(raw_fn).parameters.keys())
            args_of_raw_fn = dict(zip(argnames_of_raw_fn[:len(args)], args))
            args_of_raw_fn.update(kwargs)
            # Format resources paths if needed.
            resources_new = []
            for resource in resources:
                path = resource.path.format(**args_of_raw_fn)
                resource = type(resource)(path)
                resources_new.append(resource)

            # Try to load raw function results from cached resources,
            # otherwise eveluate the function and save its results to cache.
            try:
                results = [r.load() for r in resources_new]
                logger.info("Loaded all from cache.")
            except:
                logger.info("Cannot load from cache, evaluating.")
                results = raw_fn(*args, **kwargs)
                if len(resources_new) == 1:
                    results = [results]
                for result, r in zip(results, resources_new):
                    r.save(result)

            # If raw functin has only one result, unwrap it
            # and return as single unit.
            if len(resources_new) == 1:
                results = results[0]

            return results
        return new_fn
    return decorator
