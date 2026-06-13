import unittest
import pandas as pd
import numpy as np
from data_binning import DataBinning


class TestDataBinning(unittest.TestCase):

    def setUp(self):
        self.binner = DataBinning()
        self.test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.test_data_series = pd.Series(self.test_data)
        self.test_data_array = np.array(self.test_data)

    def test_equal_width_binning_with_list(self):
        result, intervals = self.binner.equal_width_binning(self.test_data, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_width_binning_with_series(self):
        result, intervals = self.binner.equal_width_binning(self.test_data_series, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_width_binning_with_array(self):
        result, intervals = self.binner.equal_width_binning(self.test_data_array, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_width_binning_intervals(self):
        result, intervals = self.binner.equal_width_binning(self.test_data, bins=5, return_intervals=True)
        widths = [interval.right - interval.left for interval in intervals]
        expected_width = (max(self.test_data) - min(self.test_data)) / 5
        for width in widths:
            self.assertAlmostEqual(width, expected_width, places=1)

    def test_equal_width_binning_with_labels(self):
        labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        result, intervals = self.binner.equal_width_binning(
            self.test_data, bins=5, labels=labels, return_intervals=True
        )
        self.assertEqual(result.iloc[0], 'Low')
        self.assertEqual(result.iloc[9], 'High')

    def test_equal_width_binning_invalid_bins(self):
        with self.assertRaises(ValueError):
            self.binner.equal_width_binning(self.test_data, bins=0)
        with self.assertRaises(ValueError):
            self.binner.equal_width_binning(self.test_data, bins=-1)

    def test_equal_width_binning_empty_data(self):
        with self.assertRaises(ValueError):
            self.binner.equal_width_binning([], bins=5)

    def test_equal_frequency_binning_with_list(self):
        result, intervals = self.binner.equal_frequency_binning(self.test_data, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_frequency_binning_with_series(self):
        result, intervals = self.binner.equal_frequency_binning(self.test_data_series, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_frequency_binning_with_array(self):
        result, intervals = self.binner.equal_frequency_binning(self.test_data_array, bins=5, return_intervals=True)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(intervals), 5)

    def test_equal_frequency_binning_counts(self):
        result, intervals = self.binner.equal_frequency_binning(self.test_data, bins=5, return_intervals=True)
        bin_counts = result.value_counts().sort_index()
        expected_count = len(self.test_data) // 5
        for count in bin_counts:
            self.assertTrue(count in [expected_count, expected_count + 1])

    def test_equal_frequency_binning_with_labels(self):
        labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        result, intervals = self.binner.equal_frequency_binning(
            self.test_data, bins=5, labels=labels, return_intervals=True
        )
        self.assertEqual(result.iloc[0], 'Q1')
        self.assertEqual(result.iloc[9], 'Q5')

    def test_equal_frequency_binning_invalid_bins(self):
        with self.assertRaises(ValueError):
            self.binner.equal_frequency_binning(self.test_data, bins=0)
        with self.assertRaises(ValueError):
            self.binner.equal_frequency_binning(self.test_data, bins=-1)
        with self.assertRaises(ValueError):
            self.binner.equal_frequency_binning(self.test_data, bins=100)

    def test_equal_frequency_binning_empty_data(self):
        with self.assertRaises(ValueError):
            self.binner.equal_frequency_binning([], bins=5)

    def test_get_bin_stats(self):
        result = self.binner.equal_width_binning(self.test_data, bins=5)
        stats = self.binner.get_bin_stats(self.test_data, result)
        self.assertEqual(len(stats), 5)
        self.assertIn('count', stats.columns)
        self.assertIn('mean', stats.columns)
        self.assertIn('frequency', stats.columns)
        self.assertAlmostEqual(stats['frequency'].sum(), 1.0, places=5)

    def test_bin_to_dict(self):
        result = self.binner.equal_width_binning(self.test_data, bins=5)
        bin_dict = self.binner.bin_to_dict(self.test_data, result)
        self.assertEqual(len(bin_dict), 5)
        all_values = []
        for values in bin_dict.values():
            all_values.extend(values)
        self.assertEqual(sorted(all_values), sorted(self.test_data))

    def test_equal_width_vs_equal_frequency(self):
        data = [1, 1, 1, 1, 1, 2, 3, 100, 101, 102]

        width_result = self.binner.equal_width_binning(data, bins=2)
        width_counts = width_result.value_counts().sort_index().tolist()

        freq_result = self.binner.equal_frequency_binning(data, bins=2)
        freq_counts = freq_result.value_counts().sort_index().tolist()

        self.assertNotEqual(width_counts, freq_counts)
        self.assertEqual(freq_counts, [5, 5])
        self.assertEqual(width_counts, [7, 3])


if __name__ == '__main__':
    unittest.main()
