"""
コール件数予測モデルのためのデータ準備スクリプト
- 各データファイルのマージ
- 欠損値確認と処理
"""

import pandas as pd
import numpy as np

# データファイルのパス
DATA_DIR = "/home/user/gci_2025_winter/AirREGI"

# 1. データの読み込み
print("=" * 60)
print("1. データの読み込み")
print("=" * 60)

# コール件数（目的変数）
call_df = pd.read_csv(f"{DATA_DIR}/regi_call_data_transform.csv")
call_df['cdr_date'] = pd.to_datetime(call_df['cdr_date'])
print(f"\n[コール件数] regi_call_data_transform.csv")
print(f"  期間: {call_df['cdr_date'].min()} ～ {call_df['cdr_date'].max()}")
print(f"  行数: {len(call_df)}")

# アカウント取得数
acc_df = pd.read_csv(f"{DATA_DIR}/regi_acc_get_data_transform.csv")
acc_df['cdr_date'] = pd.to_datetime(acc_df['cdr_date'])
print(f"\n[アカウント取得数] regi_acc_get_data_transform.csv")
print(f"  期間: {acc_df['cdr_date'].min()} ～ {acc_df['cdr_date'].max()}")
print(f"  行数: {len(acc_df)}")

# カレンダーデータ
cal_df = pd.read_csv(f"{DATA_DIR}/calender_data.csv")
cal_df['cdr_date'] = pd.to_datetime(cal_df['cdr_date'])
# holiday_nameのNAを空文字に変換（祝日名は無視するため）
cal_df['holiday_name'] = cal_df['holiday_name'].replace('NA', np.nan)
print(f"\n[カレンダー] calender_data.csv")
print(f"  期間: {cal_df['cdr_date'].min()} ～ {cal_df['cdr_date'].max()}")
print(f"  行数: {len(cal_df)}")

# キャンペーンデータ
cm_df = pd.read_csv(f"{DATA_DIR}/cm_data.csv")
cm_df['cdr_date'] = pd.to_datetime(cm_df['cdr_date'])
print(f"\n[キャンペーン] cm_data.csv")
print(f"  期間: {cm_df['cdr_date'].min()} ～ {cm_df['cdr_date'].max()}")
print(f"  行数: {len(cm_df)}")

# Google Trendsデータ（週次）
gt_df = pd.read_csv(f"{DATA_DIR}/gt_service_name.csv")
gt_df['week'] = pd.to_datetime(gt_df['week'])
print(f"\n[Google Trends（週次）] gt_service_name.csv")
print(f"  期間: {gt_df['week'].min()} ～ {gt_df['week'].max()}")
print(f"  行数: {len(gt_df)}")

# 2. Google Trendsデータを日次に変換
print("\n" + "=" * 60)
print("2. Google Trendsデータを日次に変換")
print("=" * 60)

# 週の開始日から7日間に同じ値を割り当てる
gt_daily_records = []
for _, row in gt_df.iterrows():
    week_start = row['week']
    search_cnt = row['search_cnt']
    for i in range(7):
        date = week_start + pd.Timedelta(days=i)
        gt_daily_records.append({'cdr_date': date, 'search_cnt': search_cnt})

gt_daily_df = pd.DataFrame(gt_daily_records)
print(f"  週次データを日次に展開: {len(gt_df)}行 → {len(gt_daily_df)}行")

# 3. データのマージ
print("\n" + "=" * 60)
print("3. データのマージ")
print("=" * 60)

# コール件数をベースにしてマージ（left join）
merged_df = call_df.copy()
print(f"\n  ベース（コール件数）: {len(merged_df)}行")

# カレンダーデータをマージ
merged_df = merged_df.merge(cal_df.drop('holiday_name', axis=1), on='cdr_date', how='left')
print(f"  + カレンダー: {len(merged_df)}行")

# アカウント取得数をマージ
merged_df = merged_df.merge(acc_df, on='cdr_date', how='left')
print(f"  + アカウント取得数: {len(merged_df)}行")

# キャンペーンデータをマージ
merged_df = merged_df.merge(cm_df, on='cdr_date', how='left')
print(f"  + キャンペーン: {len(merged_df)}行")

# Google Trendsをマージ
merged_df = merged_df.merge(gt_daily_df, on='cdr_date', how='left')
print(f"  + Google Trends: {len(merged_df)}行")

# 4. 欠損値の確認
print("\n" + "=" * 60)
print("4. 欠損値の確認")
print("=" * 60)

print("\n[各カラムの欠損値数]")
missing_counts = merged_df.isnull().sum()
for col, count in missing_counts.items():
    if count > 0:
        print(f"  {col}: {count}件")

if missing_counts.sum() == 0:
    print("  欠損値なし")

# 5. 数値データの欠損がある日付と曜日を特定
print("\n" + "=" * 60)
print("5. 数値データの欠損がある日付と曜日")
print("=" * 60)

numeric_cols = ['call_num', 'dow', 'woy', 'wom', 'doy', 'financial_year',
                'acc_get_cnt', 'cm_flg', 'search_cnt']

for col in numeric_cols:
    if col in merged_df.columns:
        missing_mask = merged_df[col].isnull()
        if missing_mask.any():
            missing_rows = merged_df[missing_mask][['cdr_date', 'dow_name']].copy()
            if 'dow_name' not in missing_rows.columns or missing_rows['dow_name'].isnull().all():
                # 曜日名がない場合は日付から計算
                missing_rows['dow_name'] = missing_rows['cdr_date'].dt.day_name()
            print(f"\n[{col}] 欠損: {len(missing_rows)}件")
            print(missing_rows.to_string(index=False))

# 6. 欠損値の処理
print("\n" + "=" * 60)
print("6. 欠損値の処理")
print("=" * 60)

# 欠損がある場合のみ処理
if merged_df.isnull().any().any():
    print("\n  欠損値を前方補完（ffill）で処理...")
    merged_df_filled = merged_df.fillna(method='ffill')

    # 先頭に欠損が残っている場合は後方補完
    merged_df_filled = merged_df_filled.fillna(method='bfill')

    print("  処理後の欠損値数:")
    remaining_missing = merged_df_filled.isnull().sum()
    for col, count in remaining_missing.items():
        if count > 0:
            print(f"    {col}: {count}件")
    if remaining_missing.sum() == 0:
        print("    欠損値なし")

    merged_df = merged_df_filled
else:
    print("  欠損値なし - 処理不要")

# 7. 最終データの確認
print("\n" + "=" * 60)
print("7. 最終データの確認")
print("=" * 60)

print(f"\n[データ形状] {merged_df.shape}")
print(f"\n[カラム一覧]")
for col in merged_df.columns:
    dtype = merged_df[col].dtype
    print(f"  - {col}: {dtype}")

print(f"\n[データ概要]")
print(merged_df.describe())

print(f"\n[先頭5行]")
print(merged_df.head())

print(f"\n[末尾5行]")
print(merged_df.tail())

# 8. CSVファイルに保存
print("\n" + "=" * 60)
print("8. マージ済みデータの保存")
print("=" * 60)

output_path = f"{DATA_DIR}/merged_call_data.csv"
merged_df.to_csv(output_path, index=False)
print(f"  保存先: {output_path}")
print(f"  行数: {len(merged_df)}, カラム数: {len(merged_df.columns)}")

# holiday_flagのTRUE/FALSEを数値に変換したバージョンも保存
merged_df_numeric = merged_df.copy()
merged_df_numeric['day_before_holiday_flag'] = merged_df_numeric['day_before_holiday_flag'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
merged_df_numeric['holiday_flag'] = merged_df_numeric['holiday_flag'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})

output_path_numeric = f"{DATA_DIR}/merged_call_data_numeric.csv"
merged_df_numeric.to_csv(output_path_numeric, index=False)
print(f"  保存先（フラグを数値化）: {output_path_numeric}")

print("\n" + "=" * 60)
print("処理完了")
print("=" * 60)
