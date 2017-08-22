import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from targets import CSVTarget, ShapefileTarget, BcolzTarget, PickleTarget as Target


@pytest.fixture
def df():
    df = pd.DataFrame(np.random.rand(100, 5))
    # Convert RangeIndex to regular index and numbers to strings
    df.columns = [str(i) for i in df.columns]
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
    with tempfile.NamedTemporaryFile() as f:
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


def test_target_save_load(df, tmpfile):
    target = Target(tmpfile)
    target.save(df)
    loaded_df = target.load()

    assert (df == loaded_df).values.all()


def test_target_one_saves_another_loads(df, tmpfile):
    one = Target(tmpfile)
    another = Target(tmpfile)
    one.save(df)
    loaded_df = another.load()

    assert (df == loaded_df).values.all()


def test_csvtarget_saves_pandas_loads(df, tmpfile):
    target = CSVTarget(tmpfile)
    target.save(df)
    loaded_df = pd.read_csv(tmpfile)

    _compare_dfs(df, loaded_df)


def test_pandas_saves_csvtarget_loads(df, tmpfile):
    kwargs = dict(sep=';', encoding='cp1251')
    target = CSVTarget(tmpfile, **kwargs)
    df.to_csv(tmpfile, **kwargs, index=False)
    loaded_df = target.load()

    _compare_dfs(df, loaded_df)


def test_shapefiletarget_saves_geopandas_loads(df, points, tmpfile):
    gdf = gpd.GeoDataFrame(df.assign(geometry=points))

    target = ShapefileTarget(tmpfile)
    target.save(gdf)
    loaded_gdf = gpd.read_file(tmpfile)

    _compare_dfs(gdf, loaded_gdf)
