import os
import pytest
from tempfile import NamedTemporaryFile
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from resources import CSV, Shapefile, Bcolz, Pickle as Resource, cache


@pytest.fixture
def df():
    df = pd.DataFrame(np.random.rand(100, 5))
    # Convert RangeIndex to regular index and numbers to strings
    df.columns = ['column' + str(i) for i in df.columns]
    return df

@pytest.fixture
def points():
    coords = np.random.randn(100, 2).cumsum(axis=1)
    points = gpd.GeoSeries([
        shapely.geometry.Point([x, y])
        for x, y in coords
    ])
    return points

@pytest.fixture
def tmpfile():
    with NamedTemporaryFile() as f:
        name = f.name
    return name


def _compare_dfs(df1, df2, eps=1e-12):
    assert df1.shape == df2.shape
    assert (df1.columns == df2.columns).all()
    for c in df1.columns:
        assert df1[c].dtype == df2[c].dtype
        if df1[c].dtype == float:
            assert (np.abs((df1[c] - df2[c]).values) < eps).all()
        else:
            assert (df1[c] == df2[c]).all()


def test_resource_save_load(df, tmpfile):
    resource = Resource(tmpfile)
    resource.save(df)
    loaded_df = resource.load()

    assert (df == loaded_df).values.all()


def test_resource_one_saves_another_loads(df, tmpfile):
    one = Resource(tmpfile)
    another = Resource(tmpfile)
    one.save(df)
    loaded_df = another.load()

    assert (df == loaded_df).values.all()


def test_csvresource_saves_pandas_loads(df, tmpfile):
    resource = CSV(tmpfile)
    resource.save(df)
    loaded_df = pd.read_csv(tmpfile)

    _compare_dfs(df, loaded_df)


def test_pandas_saves_csvresource_loads(df, tmpfile):
    kwargs = dict(sep=';', encoding='cp1251')
    resource = CSV(tmpfile, **kwargs)
    df.to_csv(tmpfile, **kwargs, index=False)
    loaded_df = resource.load()

    _compare_dfs(df, loaded_df)


def test_shapefileresource_saves_geopandas_loads(df, points, tmpfile):
    gdf = gpd.GeoDataFrame(df.assign(geometry=points))

    resource = Shapefile(tmpfile)
    resource.save(gdf)
    loaded_gdf = gpd.read_file(tmpfile)

    _compare_dfs(gdf, loaded_gdf)


# def test_bcolzresource_saves_geodataframe(df, points, tmpfile):
#     gdf = gpd.GeoDataFrame(df.assign(geometry=points))

#     resource = Bcolz(tmpfile)
#     resource.save(gdf)
#     loaded_gdf = resource.load()

#     _compare_dfs(gdf, loaded_gdf)

def test_cache_saves_and_loads(df, tmpfile):
    resource = Resource(tmpfile)

    @cache(resource)
    def process_df(df):
        df2 = 10 * df
        return df2

    df2 = process_df(df)
    df2_from_cache = process_df(df)

    _compare_dfs(df2, df2_from_cache)


def test_cache_formats_paths_from_function_args(df, tmpfile):
    # Create resource template (path contains substitutions).
    mult = 100
    resource_tpl = Resource(tmpfile + '_{mult}')

    # Define a function with parametrized cacher.
    @cache(resource_tpl)
    def process_df(df, mult):
        df2 = mult * df
        return df2

    # Nothing changes in the resource template after function call.
    df2 = process_df(df, mult)
    assert resource_tpl.path.endswith('_{mult}')

    # Cacher loads data from the parametrized cache.
    df2_from_cache = process_df(df, mult)
    _compare_dfs(df2, df2_from_cache)

    # Handcrafted resource opens and contains the same data.
    resource = Resource(tmpfile + '_{}'.format(mult))
    df2_from_resource = resource.load()
    _compare_dfs(df2, df2_from_resource)


def test_cache_formats_paths_from_object_properties(df, tmpfile):
    # Create resource template (path contains substitutions).
    add = 29
    mult = 7

    # Define class with a parameter.
    class Processor:
        def __init__(self, add):
            self._add = add

        # Define a method with a cacher parametrized by class's property.
        @cache(Resource(tmpfile + '_{mult}_{self._add}'))
        def process_df(self, df, mult):
            df2 = mult * df + self._add
            return df2

    processor = Processor(add)

    # Cacher loads data from the parametrized cache.
    df2 = processor.process_df(df, mult)
    df2_from_cache = processor.process_df(df, mult)
    _compare_dfs(df2, df2_from_cache)

    # Handcrafted resource opens and contains the same data.
    resource = Resource(tmpfile + '_%s_%s' % (mult, add))
    df2_from_resource = resource.load()
    _compare_dfs(df2, df2_from_resource)


def test_cache_works_faster_than_recomputing(df, tmpfile):
    resource = Resource(tmpfile)
    delay = 5

    @cache(resource)
    def process_df(df):
        print("Make long computation")
        time.sleep(delay)
        df2 = 10 * df
        return df2

    t0 = time.time()
    df2 = process_df(df)
    t1 = time.time()
    df2_from_cache = process_df(df)
    t2 = time.time()

    assert (t1 - t0) >= delay
    assert (t1 - t0) > (t2 - t1)


def test_cache_many_arg_func(df, tmpfile):
    resource1 = Resource(tmpfile + '1')
    resource2 = Resource(tmpfile + '2')

    @cache(resource1, resource2)
    def process_df(df):
        df2 = 10 * df
        df3 = 20 * df
        return df2, df3

    df2, df3 = process_df(df)
    df2_from_cache, df3_from_cache = process_df(df)

    _compare_dfs(df2, df2_from_cache)
    _compare_dfs(df3, df3_from_cache)

