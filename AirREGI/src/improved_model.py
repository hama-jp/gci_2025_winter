"""
改善版モデル: データプーリングと拡張特徴量

問題点:
- 各曜日の訓練データが少なすぎる（7-8サンプル）
- 過学習が発生

改善策:
- 全曜日のデータをプールして学習
- 曜日をダミー変数として扱う
- ベースライン比率を目的変数に
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_cleaning import load_and_clean_data

print("=" * 70)
print("改善版モデル: データプーリング")
print("=" * 70)

# データ読み込み
df_all, df_business = load_and_clean_data(verbose=False)

df = df_business.copy()
df['year'] = df['cdr_date'].dt.year
df['week'] = df['cdr_date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
df['month'] = df['cdr_date'].dt.month

# =============================================================================
# 特徴量作成
# =============================================================================
# 曜日別のラグ
lag_by_dow = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10}
for dow in [1, 2, 3, 4, 5]:
    lag = lag_by_dow[dow]
    df.loc[df['dow'] == dow, 'baseline_lag'] = df.loc[df['dow'] == dow, 'call_num'].shift(lag)
    df.loc[df['dow'] == dow, 'ma5'] = df.loc[df['dow'] == dow, 'call_num'].shift(lag).rolling(5, min_periods=1).mean()
    df.loc[df['dow'] == dow, 'ma10'] = df.loc[df['dow'] == dow, 'call_num'].shift(lag).rolling(10, min_periods=1).mean()

# 平日のみ抽出
df = df[df['dow'].isin([1, 2, 3, 4, 5])].copy()

# 曜日ダミー
for dow in [2, 3, 4, 5]:  # 月曜日を基準
    df[f'dow_{dow}'] = (df['dow'] == dow).astype(int)

# 外部変数（全データ共通のラグ5）
df['acc_lag'] = df['acc_get_cnt'].shift(5)
df['search_lag'] = df['search_cnt'].shift(5)

# 目的変数: ベースラインからの比率
df['target_ratio'] = df['call_num'] / df['baseline_lag']

unique_weeks = sorted(df['year_week'].unique())
start_idx = 10
n_weeks = 30
dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}

# =============================================================================
# 方法1: 比率予測モデル（全曜日プール）
# =============================================================================
print("\n" + "=" * 70)
print("方法1: 比率予測モデル（全曜日データをプール）")
print("=" * 70)

feature_sets = {
    'minimal': ['dow_2', 'dow_3', 'dow_4', 'dow_5'],
    'with_month': ['dow_2', 'dow_3', 'dow_4', 'dow_5', 'month'],
    'with_cm': ['dow_2', 'dow_3', 'dow_4', 'dow_5', 'cm_flg'],
    'full': ['dow_2', 'dow_3', 'dow_4', 'dow_5', 'month', 'cm_flg'],
}

for name, features in feature_sets.items():
    all_results = []

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        # 訓練データ（全曜日プール）
        train_mask = df['year_week'].isin(train_weeks)
        train_df = df[train_mask].dropna(subset=['baseline_lag', 'target_ratio'] + features)

        if len(train_df) < 30:
            continue

        # テストデータ
        test_mask = df['year_week'] == test_week
        test_df = df[test_mask].dropna(subset=['baseline_lag', 'target_ratio'] + features)

        if len(test_df) == 0:
            continue

        X_train = train_df[features]
        y_train = train_df['target_ratio']
        X_test = test_df[features]

        # モデル学習
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        pred_ratio = model.predict(X_test)

        # 予測値 = ベースライン × 予測比率
        preds = test_df['baseline_lag'].values * pred_ratio
        actuals = test_df['call_num'].values
        baselines = test_df['baseline_lag'].values

        mae = mean_absolute_error(actuals, preds)
        baseline_mae = mean_absolute_error(actuals, baselines)

        all_results.append({
            'week': test_week,
            'mae': mae,
            'baseline_mae': baseline_mae,
            'actual_mean': np.mean(actuals)
        })

    results_df = pd.DataFrame(all_results)
    avg_mae = results_df['mae'].mean()
    avg_baseline = results_df['baseline_mae'].mean()
    improvement = (avg_baseline - avg_mae) / avg_baseline * 100

    print(f"  {name}: MAE={avg_mae:.1f}件 (改善率: {improvement:+.1f}%)")


# =============================================================================
# 方法2: 直接予測（ベースラインを特徴量として使用）
# =============================================================================
print("\n" + "=" * 70)
print("方法2: 直接予測モデル（ベースラインを特徴量として使用）")
print("=" * 70)

feature_sets_direct = {
    'baseline_only': ['baseline_lag'],
    'baseline+dow': ['baseline_lag', 'dow_2', 'dow_3', 'dow_4', 'dow_5'],
    'baseline+dow+cm': ['baseline_lag', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'cm_flg'],
    'baseline+ma': ['baseline_lag', 'ma5', 'ma10'],
    'full': ['baseline_lag', 'ma5', 'ma10', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'cm_flg'],
}

for name, features in feature_sets_direct.items():
    all_results = []

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        # 訓練データ
        train_mask = df['year_week'].isin(train_weeks)
        train_df = df[train_mask].dropna(subset=['call_num'] + features)

        if len(train_df) < 30:
            continue

        # テストデータ
        test_mask = df['year_week'] == test_week
        test_df = df[test_mask].dropna(subset=['call_num'] + features)

        if len(test_df) == 0:
            continue

        X_train = train_df[features]
        y_train = train_df['call_num']
        X_test = test_df[features]

        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        actuals = test_df['call_num'].values
        baselines = test_df['baseline_lag'].values

        mae = mean_absolute_error(actuals, preds)
        baseline_mae = mean_absolute_error(actuals, baselines)

        all_results.append({
            'week': test_week,
            'mae': mae,
            'baseline_mae': baseline_mae,
            'actual_mean': np.mean(actuals)
        })

    results_df = pd.DataFrame(all_results)
    avg_mae = results_df['mae'].mean()
    avg_baseline = results_df['baseline_mae'].mean()
    improvement = (avg_baseline - avg_mae) / avg_baseline * 100

    print(f"  {name}: MAE={avg_mae:.1f}件 (改善率: {improvement:+.1f}%)")


# =============================================================================
# 方法3: 最良モデルの詳細評価
# =============================================================================
print("\n" + "=" * 70)
print("方法3: 最良モデルの詳細評価")
print("=" * 70)

# 最もシンプルで効果的な特徴量セット
features = ['baseline_lag']
all_preds_data = []

for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
    train_weeks = unique_weeks[:week_idx]
    test_week = unique_weeks[week_idx]

    train_mask = df['year_week'].isin(train_weeks)
    train_df = df[train_mask].dropna(subset=['call_num'] + features)

    test_mask = df['year_week'] == test_week
    test_df = df[test_mask].dropna(subset=['call_num'] + features)

    if len(train_df) < 30 or len(test_df) == 0:
        continue

    X_train = train_df[features]
    y_train = train_df['call_num']
    X_test = test_df[features]

    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    for i, (_, row) in enumerate(test_df.iterrows()):
        all_preds_data.append({
            'week': test_week,
            'dow': row['dow'],
            'date': row['cdr_date'],
            'actual': row['call_num'],
            'pred': preds[i],
            'baseline': row['baseline_lag'],
            'coef': model.coef_[0],
            'intercept': model.intercept_
        })

preds_df = pd.DataFrame(all_preds_data)

# 全体結果
model_mae = mean_absolute_error(preds_df['actual'], preds_df['pred'])
baseline_mae = mean_absolute_error(preds_df['actual'], preds_df['baseline'])
improvement = (baseline_mae - model_mae) / baseline_mae * 100

print(f"\n全体結果:")
print(f"  Ridgeモデル: MAE={model_mae:.1f}件, 誤差率={model_mae / preds_df['actual'].mean() * 100:.1f}%")
print(f"  ベースライン: MAE={baseline_mae:.1f}件, 誤差率={baseline_mae / preds_df['actual'].mean() * 100:.1f}%")
print(f"  改善率: {improvement:+.1f}%")

# 学習された係数の推移
print(f"\n学習された係数の例:")
sample_coefs = preds_df.groupby('week').first()[['coef', 'intercept']].tail(10)
for week, row in sample_coefs.iterrows():
    print(f"  {week}: coef={row['coef']:.3f}, intercept={row['intercept']:.1f}")

# 曜日別結果
print(f"\n曜日別結果:")
for dow in [1, 2, 3, 4, 5]:
    dow_data = preds_df[preds_df['dow'] == dow]
    if len(dow_data) > 0:
        dow_mae = mean_absolute_error(dow_data['actual'], dow_data['pred'])
        dow_baseline = mean_absolute_error(dow_data['actual'], dow_data['baseline'])
        dow_improvement = (dow_baseline - dow_mae) / dow_baseline * 100
        print(f"  {dow_names[dow]}曜日: MAE={dow_mae:.1f} vs ベースライン={dow_baseline:.1f} ({dow_improvement:+.1f}%)")

# =============================================================================
# サマリー
# =============================================================================
print("\n" + "=" * 70)
print("サマリー")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    改善版モデル結果                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【手法】                                                             │
│   全曜日のデータをプールして学習（訓練サンプル数: ~40→~200）         │
│   Ridge回帰（alpha=10.0）でベースラインに重み付け                    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        │  MAE      │  誤差率   │  改善率            │
├─────────────────────────────────────────────────────────────────────┤
│  Ridgeモデル            │  {model_mae:5.1f}件  │  {model_mae / preds_df['actual'].mean() * 100:5.1f}%   │  {improvement:+5.1f}%          │
│  ベースライン           │  {baseline_mae:5.1f}件  │  {baseline_mae / preds_df['actual'].mean() * 100:5.1f}%   │    0.0%          │
└─────────────────────────────────────────────────────────────────────┘
""")
