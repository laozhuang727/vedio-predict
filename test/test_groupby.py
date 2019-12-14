import unittest
import numpy as np
import pandas as pd
import vaex

from core.utils import timeit

df = pd.DataFrame({'A': [1, 1, 2, 2],
                   'B': [1, 2, 3, 4],
                   'C': np.random.randn(4)})


class MyTestCase(unittest.TestCase):
    def test_agg_min(self):
        print(df.groupby(['A'], as_index=False)['B'])

    def test_agg_by(self):
        print(df)

        print(df.groupby(['A'], as_index=False)['B'].agg({
            'groupBy:{}--aggBy:{}_max'.format('A', 'B'): 'max',
        })
        )

    def test_join(self):
        a = np.array(['a', 'b', 'c'])
        x = np.arange(1, 4)
        ds1 = vaex.from_arrays(a=a, x=x)
        print("ds1:")
        print(ds1)

        b = np.array(['a', 'b', 'd'])
        y = x ** 2
        ds2 = vaex.from_arrays(b=b, y=y)
        print("\n\n\nds2:")
        print(ds2)

        print("\n\n\njoined df:")
        print(ds1.join(ds2, left_on='a', right_on='b', how="inner"))

    def test_merge(self):
        a = np.array(['a', 'b', 'c'])
        x = np.arange(1, 4)
        ds1 = vaex.from_arrays(a=a, x=x)
        print("ds1:")
        print(ds1)

        b = np.array(['a', 'b', 'd'])
        y = x ** 2
        ds2 = vaex.from_arrays(a=a, y=y)
        print("\n\n\nds2:")
        print(ds2)

        print("\n\n\njoined df:")
        ds1 = ds1.join(ds2, on='a', how="inner", rprefix="rprefix_")
        print(ds1)

        print("delete rprefix_")
        ds1 = ds1.drop('rprefix_a')
        print(ds1)

    def test_daytime_column(self):
        date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
        df = vaex.from_arrays(date=date)
        print(df.describe())
        print(df.dtypes)


    def test_fillna(self):
        x = np.array([3, 1, np.nan, 10, np.nan])
        df = vaex.from_arrays(x=x)
        df_filled = df.fillna(value=-1, column_names=['x'])
        print(df_filled)

    def test_fillna(self):
        x = np.array([3, 1, np.nan, 10, np.nan])
        df = vaex.from_arrays(x=x)
        df['x_filled'] = df.x.fillna(value=-1)
        print(df)

        df['x'] = df['x_filled']
        df = df.drop('x_filled')
        print(df)

    def test_groupby_subset(self):
        np.random.seed(42)
        x = np.random.randint(1, 5, 10)
        y = x ** 2
        df = vaex.from_arrays(x=x, y=y)
        df = df.groupby(df.x, agg=None)
        print(df)

    @timeit
    def test_hello_print(self):
        print("12321321")

    def test_df_example_head(self):
        df = vaex.example()

        print(df.head())

    def test_df_apply(self):
        df = vaex.example()

        def func(x, y):
            return (x + y) / (x - y)

        apply_func = df.apply(func, arguments=[df.x, df.y])
        print(apply_func)

    def test_df_evaluate(self):
        df = vaex.example()

        def func(x, y):
            return (x + y) / (x - y)

        apply_func = df.apply(func, arguments=[df.x, df.y])

        df['new_col'] = df.evaluate(apply_func)
        print(df.min(df['new_col']))
        print(df.mean(df['new_col']))
        print(df.max(df['new_col']))

if __name__ == '__main__':
    unittest.main()
