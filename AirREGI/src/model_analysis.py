"""
モデル分析: なぜシンプルなRidgeが良いのか？改善の余地を探る
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from data_cleaning import load_and_clean_data

print("=" * 70)
print("モデル分析: 改善の余地を探る")
print("=" * 70)

# データ読み込み
df_all, df_business = load_and_clean_data(verbose=False)

df = df_business.copy()
df['year'] = df['cdr_date'].dt.year
df['week'] = df['cdr_date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
df['month'] = df['cdr_date'].dt.month

# ラグ作成
lag_by_dow = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10}
for dow in [1, 2, 3, 4, 5]:
    lag = lag_by_dow[dow]
    df[f'baseline_lag_{dow}'] = df['call_num'].shift(lag)
    df[f'ma5_{dow}'] = df['call_num'].shift(lag).rolling(5, min_periods=1).mean()
    df[f'ma20_{dow}'] = df['call_num'].shift(lag).rolling(20, min_periods=1).mean()

# 外部変数（水曜時点で知りえる情報）
df['acc_lag_5'] = df['acc_get_cnt'].shift(5)
df['search_lag_5'] = df['search_cnt'].shift(5)

unique_weeks = sorted(df['year_week'].unique())
start_idx = 10
n_weeks = 30
dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}

# =============================================================================
# 分析1: Ridgeが学習している係数を確認
# =============================================================================
print("\n" + "=" * 70)
print("分析1: Ridgeが学習している係数")
print("=" * 70)

for dow in [1, 2, 3, 4, 5]:
    train_mask = df['year_week'].isin(unique_weeks[:start_idx]) & (df['dow'] == dow)
    train_df = df[train_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

    X = train_df[[f'baseline_lag_{dow}']]
    y = train_df['call_num']

    lr = LinearRegression()
    lr.fit(X, y)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    print(f"\n  {dow_names[dow]}曜日 (N={len(train_df)}):")
    print(f"    LinearRegression: coef={lr.coef_[0]:.3f}, intercept={lr.intercept_:.1f}")
    print(f"    Ridge(alpha=1):   coef={ridge.coef_[0]:.3f}, intercept={ridge.intercept_:.1f}")

    # 実際のcall_numの平均と標準偏差
    print(f"    call_num: mean={y.mean():.1f}, std={y.std():.1f}")

# =============================================================================
# 分析2: ベースラインからの乖離パターン
# =============================================================================
print("\n" + "=" * 70)
print("分析2: ベースラインからの乖離パターン")
print("=" * 70)

for dow in [1, 2, 3, 4, 5]:
    valid_mask = df['dow'] == dow
    valid_df = df[valid_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

    errors = valid_df['call_num'] - valid_df[f'baseline_lag_{dow}']

    print(f"\n  {dow_names[dow]}曜日:")
    print(f"    誤差平均: {errors.mean():+.1f}件 (ベースラインが{('過大' if errors.mean() < 0 else '過小')}予測傾向)")
    print(f"    誤差標準偏差: {errors.std():.1f}件")

    # 月別の誤差傾向
    monthly_errors = valid_df.groupby('month').apply(
        lambda x: (x['call_num'] - x[f'baseline_lag_{dow}']).mean()
    )
    print(f"    月別誤差傾向: ", end="")
    for month, err in monthly_errors.items():
        print(f"{month}月:{err:+.0f} ", end="")
    print()

# =============================================================================
# 分析3: 外部変数の予測力
# =============================================================================
print("\n" + "=" * 70)
print("分析3: 外部変数（アカウント取得数）の予測力")
print("=" * 70)

# ベースラインからの残差を外部変数で予測できるか
for dow in [1, 2, 3, 4, 5]:
    valid_mask = (df['dow'] == dow) & df[[f'baseline_lag_{dow}', 'acc_lag_5', 'call_num']].notna().all(axis=1)
    valid_df = df[valid_mask].copy()

    # 残差計算
    valid_df['residual'] = valid_df['call_num'] - valid_df[f'baseline_lag_{dow}']

    # 残差とacc_lag_5の相関
    corr = valid_df['residual'].corr(valid_df['acc_lag_5'])
    print(f"  {dow_names[dow]}曜日: 残差とacc_lag_5の相関 = {corr:.3f}")

# =============================================================================
# 分析4: トレンド成分の影響
# =============================================================================
print("\n" + "=" * 70)
print("分析4: トレンド成分の影響")
print("=" * 70)

# 時間経過による変化（移動平均のトレンド）
for dow in [1, 2, 3, 4, 5]:
    valid_mask = df['dow'] == dow
    valid_df = df[valid_mask].dropna(subset=[f'baseline_lag_{dow}', f'ma20_{dow}', 'call_num'])

    # トレンド特徴量: 現在の移動平均 / ベースラインの比率
    valid_df['trend'] = valid_df[f'ma20_{dow}'] / valid_df[f'baseline_lag_{dow}']

    # トレンドと実績の相関
    corr = valid_df['call_num'].corr(valid_df['trend'])
    print(f"  {dow_names[dow]}曜日: call_numとトレンドの相関 = {corr:.3f}")

# =============================================================================
# 分析5: 曜日ごとの補正係数を試す
# =============================================================================
print("\n" + "=" * 70)
print("分析5: 曜日・月ごとの補正係数")
print("=" * 70)

# 曜日・月別の補正係数を計算
correction_factors = {}
for dow in [1, 2, 3, 4, 5]:
    valid_mask = df['dow'] == dow
    valid_df = df[valid_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

    # 月別の補正係数
    monthly_factors = valid_df.groupby('month').apply(
        lambda x: x['call_num'].mean() / x[f'baseline_lag_{dow}'].mean() if x[f'baseline_lag_{dow}'].mean() > 0 else 1.0
    )
    correction_factors[dow] = monthly_factors

print("月別補正係数 (call_num / baseline の比率):")
print("   " + "  ".join([f"{m:2d}月" for m in range(1, 13)]))
for dow in [1, 2, 3, 4, 5]:
    factors = correction_factors[dow]
    line = f"{dow_names[dow]}: "
    for m in range(1, 13):
        if m in factors.index:
            line += f"{factors[m]:.2f} "
        else:
            line += "  -  "
    print(line)

# =============================================================================
# 分析6: 補正係数を使ったモデルのテスト
# =============================================================================
print("\n" + "=" * 70)
print("分析6: 月別補正係数を使ったモデル")
print("=" * 70)

def evaluate_with_correction(use_monthly_correction=False):
    all_actuals = []
    all_preds = []
    all_baselines = []

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        for dow in [1, 2, 3, 4, 5]:
            train_mask = df['year_week'].isin(train_weeks) & (df['dow'] == dow)
            train_df = df[train_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

            test_mask = (df['year_week'] == test_week) & (df['dow'] == dow)
            test_df = df[test_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

            if len(train_df) < 5 or len(test_df) == 0:
                continue

            baseline = test_df[f'baseline_lag_{dow}'].values[0]
            actual = test_df['call_num'].values[0]

            if use_monthly_correction:
                test_month = test_df['month'].values[0]
                # 訓練データから月別補正係数を計算
                month_data = train_df[train_df['month'] == test_month]
                if len(month_data) > 0:
                    factor = month_data['call_num'].mean() / month_data[f'baseline_lag_{dow}'].mean()
                else:
                    factor = train_df['call_num'].mean() / train_df[f'baseline_lag_{dow}'].mean()
                pred = baseline * factor
            else:
                # 全体の補正係数
                factor = train_df['call_num'].mean() / train_df[f'baseline_lag_{dow}'].mean()
                pred = baseline * factor

            all_actuals.append(actual)
            all_preds.append(pred)
            all_baselines.append(baseline)

    mae = mean_absolute_error(all_actuals, all_preds)
    baseline_mae = mean_absolute_error(all_actuals, all_baselines)
    return mae, baseline_mae

mae_global, baseline_mae = evaluate_with_correction(use_monthly_correction=False)
mae_monthly, _ = evaluate_with_correction(use_monthly_correction=True)

print(f"  ベースライン: MAE={baseline_mae:.1f}件")
print(f"  全体補正係数: MAE={mae_global:.1f}件 ({(baseline_mae - mae_global) / baseline_mae * 100:+.1f}%)")
print(f"  月別補正係数: MAE={mae_monthly:.1f}件 ({(baseline_mae - mae_monthly) / baseline_mae * 100:+.1f}%)")

# =============================================================================
# 分析7: キャンペーン期間の影響
# =============================================================================
print("\n" + "=" * 70)
print("分析7: キャンペーン期間の影響")
print("=" * 70)

# キャンペーン期間中のベースライン誤差
for dow in [1, 2, 3, 4, 5]:
    valid_mask = df['dow'] == dow
    valid_df = df[valid_mask].dropna(subset=[f'baseline_lag_{dow}', 'call_num'])

    cm_errors = valid_df[valid_df['cm_flg'] == 1]['call_num'] - valid_df[valid_df['cm_flg'] == 1][f'baseline_lag_{dow}']
    no_cm_errors = valid_df[valid_df['cm_flg'] == 0]['call_num'] - valid_df[valid_df['cm_flg'] == 0][f'baseline_lag_{dow}']

    if len(cm_errors) > 0 and len(no_cm_errors) > 0:
        print(f"  {dow_names[dow]}曜日: CM期間誤差={cm_errors.mean():+.1f} (N={len(cm_errors)}), 通常期間誤差={no_cm_errors.mean():+.1f} (N={len(no_cm_errors)})")

# =============================================================================
# 結論
# =============================================================================
print("\n" + "=" * 70)
print("結論と改善の方向性")
print("=" * 70)
print("""
【発見】
1. Ridgeモデルは係数≈1.0を学習 → ベースラインに近い予測
2. 曜日・月によってベースラインからの乖離パターンが異なる
3. キャンペーン期間中はベースラインが過小予測傾向

【改善の方向性】
1. 曜日別・月別の補正係数を適用
2. キャンペーンフラグによる補正
3. トレンド成分（移動平均）による補正
4. 残差と相関のある外部変数を探す
""")
