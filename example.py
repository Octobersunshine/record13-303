import pandas as pd
import numpy as np
from data_binning import DataBinning

if __name__ == '__main__':
    binner = DataBinning()

    np.random.seed(42)
    data = np.random.normal(100, 20, 1000)

    print("=" * 60)
    print("数据分箱服务示例")
    print("=" * 60)
    print(f"原始数据量: {len(data)} 条")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"数据均值: {data.mean():.2f}")
    print()

    print("-" * 60)
    print("1. 等宽分箱示例 (Equal Width Binning)")
    print("-" * 60)
    width_result, width_intervals = binner.equal_width_binning(data, bins=5, return_intervals=True)
    print("分箱区间:")
    for interval in width_intervals:
        print(f"  {interval}")
    print()
    print("分箱统计:")
    width_stats = binner.get_bin_stats(data, width_result)
    print(width_stats.to_string(index=False))
    print()

    print("-" * 60)
    print("2. 等频分箱示例 (Equal Frequency Binning)")
    print("-" * 60)
    freq_result, freq_intervals = binner.equal_frequency_binning(data, bins=5, return_intervals=True)
    print("分箱区间:")
    for interval in freq_intervals:
        print(f"  {interval}")
    print()
    print("分箱统计:")
    freq_stats = binner.get_bin_stats(data, freq_result)
    print(freq_stats.to_string(index=False))
    print()

    print("-" * 60)
    print("3. 自定义标签的等宽分箱")
    print("-" * 60)
    labels = ['很低', '较低', '中等', '较高', '很高']
    labeled_result = binner.equal_width_binning(data, bins=5, labels=labels)
    print("分箱结果分布:")
    print(labeled_result.value_counts().sort_index())
    print()

    print("-" * 60)
    print("4. 分箱结果转字典")
    print("-" * 60)
    bin_dict = binner.bin_to_dict(data, freq_result)
    for bin_name, values in bin_dict.items():
        print(f"{bin_name}: {len(values)} 条数据, 示例值: {sorted(values)[:3]}")

    print()
    print("=" * 60)
    print("5. 等宽分箱 vs 等频分箱对比")
    print("=" * 60)
    skewed_data = [1, 1, 2, 2, 3, 3, 4, 100, 101, 102]

    width_result = binner.equal_width_binning(skewed_data, bins=2, labels=['低', '高'])
    freq_result = binner.equal_frequency_binning(skewed_data, bins=2, labels=['低', '高'])

    print(f"偏态数据: {skewed_data}")
    print()
    print("分箱结果对比:")
    print(pd.DataFrame({
        '数值': skewed_data,
        '等宽分箱': width_result,
        '等频分箱': freq_result
    }).to_string(index=False))
    print()
    print("分布对比:")
    compare_df = pd.DataFrame({
        '等宽分箱': width_result.value_counts().sort_index(),
        '等频分箱': freq_result.value_counts().sort_index()
    })
    print(compare_df.to_string())
    print()
    print("说明:")
    print("  等宽分箱: 按值域均分, 区间宽度相同, 但数据分布可能不均")
    print("  等频分箱: 按数量均分, 每箱数据量相同, 但区间宽度可能不同")
