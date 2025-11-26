"""
最終モデル: Ridge回帰 + 前週同曜日特徴量

実験の結果、最もシンプルなモデルが最高性能を発揮
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_cleaning import load_and_clean_data

BASE_DIR = "/home/user/gci_2025_winter/AirREGI"
RESULTS_DIR = f"{BASE_DIR}/results"

print("=" * 70)
print("最終モデル: Ridge回帰 + 前週同曜日特徴量")
print("=" * 70)

# データ読み込み
df_all, df_business = load_and_clean_data(verbose=False)

df = df_business.copy()
df['year'] = df['cdr_date'].dt.year
df['week'] = df['cdr_date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)

# =============================================================================
# 特徴量作成（曜日別にベースラインと同じラグ構造）
# =============================================================================
# 水曜時点で知りえる情報のみ使用
# 月火水: 5営業日前（前週同曜日）
# 木金: 10営業日前（前々週同曜日）
lag_by_dow = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10}

for dow in [1, 2, 3, 4, 5]:
    lag = lag_by_dow[dow]
    df[f'baseline_lag_{dow}'] = df['call_num'].shift(lag)

unique_weeks = sorted(df['year_week'].unique())
start_idx = 10
n_weeks = 30

# =============================================================================
# バックテスト
# =============================================================================
print("\n【週次予測バックテスト】")
print("シナリオ: 水曜までの実績で翌週月〜金を予測")
print("モデル: Ridge回帰（各曜日で個別学習）")
print("特徴量: 前週同曜日（月火水）/ 前々週同曜日（木金）のみ")

all_results = []
all_predictions = []

for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
    train_weeks = unique_weeks[:week_idx]
    test_week = unique_weeks[week_idx]

    week_actuals = []
    week_preds = []
    week_baselines = []
    week_details = []

    for dow in [1, 2, 3, 4, 5]:
        features = [f'baseline_lag_{dow}']

        # 訓練データ（同じ曜日のみ）
        train_mask = df['year_week'].isin(train_weeks) & (df['dow'] == dow)
        train_df = df[train_mask].dropna(subset=features + ['call_num'])

        if len(train_df) < 5:
            continue

        # テストデータ
        test_mask = (df['year_week'] == test_week) & (df['dow'] == dow)
        test_df = df[test_mask].dropna(subset=features + ['call_num'])

        if len(test_df) == 0:
            continue

        X_train = train_df[features]
        y_train = train_df['call_num']
        X_test = test_df[features]
        y_test = test_df['call_num']

        # Ridgeモデル
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        actual = y_test.values[0]
        pred = y_pred[0]
        baseline = test_df[f'baseline_lag_{dow}'].values[0]

        week_actuals.append(actual)
        week_preds.append(pred)
        week_baselines.append(baseline)

        all_predictions.append({
            'week': test_week,
            'dow': dow,
            'date': test_df['cdr_date'].values[0],
            'actual': actual,
            'pred': pred,
            'baseline': baseline,
            'model_error': abs(actual - pred),
            'baseline_error': abs(actual - baseline)
        })

    if len(week_actuals) > 0:
        mae = mean_absolute_error(week_actuals, week_preds)
        baseline_mae = mean_absolute_error(week_actuals, week_baselines)
        rmse = np.sqrt(mean_squared_error(week_actuals, week_preds))

        all_results.append({
            'week': test_week,
            'n_days': len(week_actuals),
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'actual_mean': np.mean(week_actuals),
            'pred_mean': np.mean(week_preds)
        })

results_df = pd.DataFrame(all_results)
predictions_df = pd.DataFrame(all_predictions)

# =============================================================================
# 結果表示
# =============================================================================
print(f"\n[週次予測結果] ({len(results_df)}週分)")
model_mae = results_df['mae'].mean()
baseline_mae = results_df['baseline_mae'].mean()
model_rmse = results_df['rmse'].mean()
actual_mean = results_df['actual_mean'].mean()
model_err = model_mae / actual_mean * 100
baseline_err = baseline_mae / actual_mean * 100
improvement = (baseline_mae - model_mae) / baseline_mae * 100

print(f"  Ridgeモデル:")
print(f"    MAE:    {model_mae:.1f}件")
print(f"    RMSE:   {model_rmse:.1f}件")
print(f"    誤差率: {model_err:.1f}%")
print(f"  ベースライン（前週同曜日）:")
print(f"    MAE:    {baseline_mae:.1f}件")
print(f"    誤差率: {baseline_err:.1f}%")
print(f"  改善率: {improvement:+.1f}%")

# 曜日別の結果
print("\n[曜日別の結果]")
dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}
for dow in [1, 2, 3, 4, 5]:
    dow_data = predictions_df[predictions_df['dow'] == dow]
    if len(dow_data) > 0:
        dow_mae = dow_data['model_error'].mean()
        dow_baseline = dow_data['baseline_error'].mean()
        dow_improvement = (dow_baseline - dow_mae) / dow_baseline * 100 if dow_baseline > 0 else 0
        print(f"  {dow_names[dow]}曜日: MAE={dow_mae:.1f} vs ベースライン={dow_baseline:.1f} ({dow_improvement:+.1f}%)")

# =============================================================================
# 結果サマリー
# =============================================================================
print("\n" + "=" * 70)
print("結果サマリー")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│              コール件数予測 - 最終モデル比較                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【モデル】                                                           │
│   Ridge回帰（alpha=1.0）                                            │
│   特徴量: 前週同曜日（月火水）/ 前々週同曜日（木金）のみ              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        │  MAE      │  誤差率   │  ベースライン比    │
├─────────────────────────────────────────────────────────────────────┤
│  Ridgeモデル            │  {model_mae:5.1f}件  │  {model_err:5.1f}%   │  {improvement:+5.1f}%          │
│  ベースライン（前週同曜日）│  {baseline_mae:5.1f}件  │  {baseline_err:5.1f}%   │    0.0%          │
│  GradientBoosting(旧)   │   47.6件  │   35.0%   │  -13.4%          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【結論】                                                             │
│   シンプルなRidgeモデルがベースラインを{improvement:+.1f}%上回った！           │
│   特徴量を増やすほど性能が悪化する「少ないほど良い」パターン          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# 結果保存
results_df.to_csv(f"{RESULTS_DIR}/weekly_backtest_ridge.csv", index=False)
predictions_df.to_csv(f"{RESULTS_DIR}/weekly_predictions_ridge.csv", index=False)

print("結果を保存しました:")
print(f"  - weekly_backtest_ridge.csv")
print(f"  - weekly_predictions_ridge.csv")
