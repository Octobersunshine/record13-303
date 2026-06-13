import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union


class DataBinning:
    def __init__(self):
        pass

    def equal_width_binning(
        self,
        data: Union[pd.Series, List[float], np.ndarray],
        bins: int,
        include_lowest: bool = True,
        right: bool = True,
        labels: Optional[List] = None,
        return_intervals: bool = False,
        return_stats: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.IntervalIndex], Tuple[pd.Series, pd.IntervalIndex, pd.DataFrame]]:
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)

        if bins <= 0:
            raise ValueError("bins must be a positive integer")

        if len(data) == 0:
            raise ValueError("data cannot be empty")

        binned_data, bin_edges = pd.cut(
            data,
            bins=bins,
            include_lowest=include_lowest,
            right=right,
            labels=labels,
            retbins=True
        )

        intervals = pd.IntervalIndex.from_breaks(bin_edges, closed='right' if right else 'left')

        if return_stats:
            stats = self.get_bin_stats(data, binned_data)
            if return_intervals:
                return binned_data, intervals, stats
            return binned_data, stats

        if return_intervals:
            return binned_data, intervals
        return binned_data

    def equal_frequency_binning(
        self,
        data: Union[pd.Series, List[float], np.ndarray],
        bins: int,
        labels: Optional[List] = None,
        duplicates: str = 'drop',
        return_intervals: bool = False,
        return_stats: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.IntervalIndex], Tuple[pd.Series, pd.IntervalIndex, pd.DataFrame]]:
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)

        if bins <= 0:
            raise ValueError("bins must be a positive integer")

        if len(data) == 0:
            raise ValueError("data cannot be empty")

        if bins > len(data):
            raise ValueError("bins cannot be larger than the number of data points")

        unique_values = pd.Series(data.unique())

        if bins > len(unique_values):
            raise ValueError(
                f"bins ({bins}) cannot be larger than the number of unique values ({len(unique_values)})"
            )

        unique_binned, bin_edges = pd.qcut(
            unique_values,
            q=bins,
            labels=labels,
            duplicates=duplicates,
            retbins=True
        )

        value_to_bin = dict(zip(unique_values, unique_binned))
        binned_data = data.map(value_to_bin)

        intervals = pd.IntervalIndex.from_breaks(bin_edges, closed='right')

        if return_stats:
            stats = self.get_bin_stats(data, binned_data)
            if return_intervals:
                return binned_data, intervals, stats
            return binned_data, stats

        if return_intervals:
            return binned_data, intervals
        return binned_data

    def get_bin_stats(
        self,
        data: Union[pd.Series, List[float], np.ndarray],
        binned_data: pd.Series
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)

        df = pd.DataFrame({'value': data, 'bin': binned_data})

        stats = df.groupby('bin', observed=True).agg(
            count=('value', 'count'),
            min=('value', 'min'),
            max=('value', 'max'),
            mean=('value', 'mean'),
            std=('value', 'std'),
            median=('value', 'median')
        ).reset_index()

        stats['frequency'] = stats['count'] / stats['count'].sum()

        def get_bin_left(x):
            if pd.notna(x) and hasattr(x, 'left'):
                return x.left
            return None

        def get_bin_right(x):
            if pd.notna(x) and hasattr(x, 'right'):
                return x.right
            return None

        def get_bin_closed(x):
            if pd.notna(x) and hasattr(x, 'closed'):
                return x.closed
            return None

        stats['bin_left'] = stats['bin'].apply(get_bin_left)
        stats['bin_right'] = stats['bin'].apply(get_bin_right)
        stats['bin_closed'] = stats['bin'].apply(get_bin_closed)

        cols = ['bin', 'bin_left', 'bin_right', 'bin_closed', 'count', 'frequency',
                'min', 'max', 'mean', 'std', 'median']
        stats = stats[cols]

        return stats

    def bin_to_dict(
        self,
        data: Union[pd.Series, List[float], np.ndarray],
        binned_data: pd.Series
    ) -> dict:
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)

        result = {}
        for bin_cat, group in pd.DataFrame({'value': data, 'bin': binned_data}).groupby('bin', observed=True):
            result[str(bin_cat)] = group['value'].tolist()

        return result
