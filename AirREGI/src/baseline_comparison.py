"""
ベースラインモデルとの比較

シンプルなベースライン:
- 前週同曜日の値をそのまま予測値とする
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_cleaning import load_and_clean_data

print("=" * 70)
print("ベースライン比較")
print("=" * 70)

# データ読み込み
df_all, df_business = load_and_clean_data(verbose=False)

# =============================================================================
# 1. シンプルベースライン: 前週同曜日
# =============================================================================
print("\n【ベースライン: 前週同曜日】")

df = df_business.copy()
df['prev_week_same_dow'] = df['call_num'].shift(5)  # 5営業日前 = 前週同曜日

# 有効なデータのみ
df_valid = df.dropna(subset=['prev_week_same_dow'])

baseline_mae = mean_absolute_error(df_valid['call_num'], df_valid['prev_week_same_dow'])
baseline_rmse = np.sqrt(mean_squared_error(df_valid['call_num'], df_valid['prev_week_same_dow']))
baseline_mean = df_valid['call_num'].mean()
baseline_error_rate = baseline_mae / baseline_mean * 100

print(f"  対象日数: {len(df_valid)}日")
print(f"  MAE: {baseline_mae:.1f}件")
print(f"  RMSE: {baseline_rmse:.1f}件")
print(f"  誤差率: {baseline_error_rate:.1f}%")

# =============================================================================
# 2. 週次予測シナリオでのベースライン
# =============================================================================
print("\n【週次予測シナリオ: 水曜時点で知りえる前週同曜日ベースライン】")
print("  月火水: 前週同曜日（5営業日前）")
print("  木金:   前々週同曜日（10営業日前）※水曜時点で未知のため")

df['year'] = df['cdr_date'].dt.year
df['week'] = df['cdr_date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)

# 曜日別のラグ（水曜時点で知りえる情報のみ）
# 月(dow=1): 前週月曜 = 5営業日前（水曜から見て2日前）
# 火(dow=2): 前週火曜 = 5営業日前（水曜から見て1日前）
# 水(dow=3): 前週水曜 = 5営業日前（水曜当日）
# 木(dow=4): 前々週木曜 = 10営業日前（前週木曜は水曜時点で未知）
# 金(dow=5): 前々週金曜 = 10営業日前（前週金曜は水曜時点で未知）
lag_by_dow = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10}

for dow, lag in lag_by_dow.items():
    df[f'baseline_dow{dow}'] = df['call_num'].shift(lag)

unique_weeks = sorted(df['year_week'].unique())
start_idx = 10
n_weeks = 30

weekly_results = []
for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
    test_week = unique_weeks[week_idx]
    test_data = df[df['year_week'] == test_week].copy()

    week_actuals = []
    week_preds = []
    for dow in [1, 2, 3, 4, 5]:
        row = test_data[test_data['dow'] == dow]
        if len(row) > 0 and not pd.isna(row[f'baseline_dow{dow}'].values[0]):
            week_actuals.append(row['call_num'].values[0])
            week_preds.append(row[f'baseline_dow{dow}'].values[0])

    if len(week_actuals) > 0:
        mae = mean_absolute_error(week_actuals, week_preds)
        weekly_results.append({
            'week': test_week,
            'mae': mae,
            'actual_mean': np.mean(week_actuals)
        })

weekly_df = pd.DataFrame(weekly_results)
weekly_baseline_mae = weekly_df['mae'].mean()
weekly_baseline_err = weekly_baseline_mae / weekly_df['actual_mean'].mean() * 100

print(f"  対象: {len(weekly_df)}週")
print(f"  MAE平均: {weekly_baseline_mae:.1f}件")
print(f"  誤差率: {weekly_baseline_err:.1f}%")

# =============================================================================
# 3. 結果サマリー
# =============================================================================
print("\n" + "=" * 70)
print("モデル比較サマリー")
print("=" * 70)

print(f"""
┌──────────────────────────────────────────────────────────┐
│                    週次予測 比較                          │
├──────────────────────────────────────────────────────────┤
│  モデル                    │  MAE    │  誤差率            │
├──────────────────────────────────────────────────────────┤
│  ベースライン（前週同曜日）  │  {weekly_baseline_mae:5.1f}件 │  {weekly_baseline_err:5.1f}%           │
│  GradientBoosting          │   47.6件 │   35.0%           │
├──────────────────────────────────────────────────────────┤
│  改善率                     │  {(weekly_baseline_mae - 47.6) / weekly_baseline_mae * 100:+5.1f}%                       │
└──────────────────────────────────────────────────────────┘
""")
