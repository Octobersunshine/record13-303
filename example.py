import pandas as pd
import numpy as np
from data_binning import DataBinning

if __name__ == '__main__':
    binner = DataBinning()

    np.random.seed(42)
    data = np.random.normal(100, 20, 1000)

    print("=" * 70)
    print("数据分箱服务示例")
    print("=" * 70)
    print(f"原始数据量: {len(data)} 条")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"数据均值: {data.mean():.2f}")
    print()

    print("-" * 70)
    print("1. 等宽分箱示例 (Equal Width Binning)")
    print("-" * 70)
    width_result, width_intervals, width_stats = binner.equal_width_binning(
        data, bins=5, return_intervals=True, return_stats=True
    )
    print("分箱统计 (含边界信息):")
    print(width_stats[['bin_left', 'bin_right', 'count', 'mean', 'frequency']].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))
    print()
    print(f"  说明: 每个区间宽度约为 {(data.max() - data.min())/5:.2f}")
    print()

    print("-" * 70)
    print("2. 等频分箱示例 (Equal Frequency Binning)")
    print("-" * 70)
    freq_result, freq_intervals, freq_stats = binner.equal_frequency_binning(
        data, bins=5, return_intervals=True, return_stats=True
    )
    print("分箱统计 (含边界信息):")
    print(freq_stats[['bin_left', 'bin_right', 'count', 'mean', 'frequency']].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))
    print()
    print(f"  说明: 每箱约 {len(data)/5:.0f} 条数据, 占比约 20%")
    print()

    print("-" * 70)
    print("3. 自定义标签的等宽分箱")
    print("-" * 70)
    labels = ['很低', '较低', '中等', '较高', '很高']
    labeled_result, labeled_stats = binner.equal_width_binning(
        data, bins=5, labels=labels, return_stats=True
    )
    print("分箱结果分布:")
    print(labeled_stats[['bin_left', 'bin_right', 'count', 'mean']].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))
    print()

    print("-" * 70)
    print("4. 分箱结果转字典")
    print("-" * 70)
    bin_dict = binner.bin_to_dict(data, freq_result)
    for bin_name, values in bin_dict.items():
        print(f"{bin_name}: {len(values)} 条数据, 示例值: {sorted(values)[:3]}")

    print()
    print("=" * 70)
    print("5. 等宽分箱 vs 等频分箱对比")
    print("=" * 70)
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
    print()

    print("=" * 70)
    print("6. 详细统计信息示例")
    print("=" * 70)
    print("等频分箱完整统计:")
    print(freq_stats[['bin_left', 'bin_right', 'bin_closed', 'count', 'frequency',
                   'min', 'max', 'mean', 'std', 'median']].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))
