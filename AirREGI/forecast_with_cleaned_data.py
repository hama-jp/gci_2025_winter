"""
コール件数予測 - クリーニング済みデータ版

変更点:
- call_num=0の日を休業日として除外（closed_flag使用）
- 営業日のみで学習・予測
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# データクリーニングモジュールをインポート
from data_cleaning import load_and_clean_data, create_features_for_prediction

DATA_DIR = "/home/user/gci_2025_winter/AirREGI"

print("=" * 70)
print("コール件数予測 - クリーニング済みデータ版")
print("=" * 70)

# =============================================================================
# 1. クリーニング済みデータの読み込み
# =============================================================================
df_all, df_business = load_and_clean_data(verbose=True)

# =============================================================================
# 2. 週次予測バックテスト
# =============================================================================
print("\n" + "=" * 70)
print("週次予測バックテスト")
print("=" * 70)
print("シナリオ: 水曜までの実績で翌週月〜金を予測")

def weekly_backtest(df, n_weeks=30):
    """週次予測のバックテスト"""
    results = []
    all_predictions = []

    df = df.copy()
    df['year'] = df['cdr_date'].dt.year
    df['week'] = df['cdr_date'].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)

    unique_weeks = sorted(df['year_week'].unique())

    # 予測対象の各曜日に対応するホライズン
    # 水曜(dow=3)から: 月(dow=1)=5日先, 火=6日先, ...
    horizons = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}

    start_idx = 10

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        week_results = []

        for target_dow, horizon in horizons.items():
            df_features, feature_cols = create_features_for_prediction(df, horizon)

            # 訓練データ
            train_mask = df_features['year_week'].isin(train_weeks)
            train_df = df_features[train_mask].dropna(subset=feature_cols + ['call_num'])

            if len(train_df) < 20:
                continue

            # テストデータ
            test_mask = (df_features['year_week'] == test_week) & (df_features['dow'] == target_dow)
            test_df = df_features[test_mask].dropna(subset=feature_cols + ['call_num'])

            if len(test_df) == 0:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df['call_num']
            X_test = test_df[feature_cols]
            y_test = test_df['call_num']

            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            week_results.append({
                'dow': target_dow,
                'actual': y_test.values[0],
                'pred': y_pred[0]
            })

            all_predictions.append({
                'week': test_week,
                'dow': target_dow,
                'date': test_df['cdr_date'].values[0],
                'actual': y_test.values[0],
                'pred': y_pred[0]
            })

        if len(week_results) > 0:
            actuals = [r['actual'] for r in week_results]
            preds = [r['pred'] for r in week_results]

            mae = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))

            results.append({
                'week': test_week,
                'n_days': len(week_results),
                'mae': mae,
                'rmse': rmse,
                'actual_mean': np.mean(actuals),
                'pred_mean': np.mean(preds)
            })

    return pd.DataFrame(results), pd.DataFrame(all_predictions)


weekly_results, weekly_predictions = weekly_backtest(df_business, n_weeks=30)

print(f"\n[週次予測結果] ({len(weekly_results)}週分)")
print(f"  MAE平均:  {weekly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {weekly_results['rmse'].mean():.1f}件")
print(f"  実績平均: {weekly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {weekly_results['pred_mean'].mean():.1f}件")
print(f"  誤差率:   {weekly_results['mae'].mean() / weekly_results['actual_mean'].mean() * 100:.1f}%")

# =============================================================================
# 3. 月次予測バックテスト
# =============================================================================
print("\n" + "=" * 70)
print("月次予測バックテスト")
print("=" * 70)
print("シナリオ: 20日までの実績で翌月1ヶ月分を予測")

def monthly_backtest(df, n_months=12):
    """月次予測のバックテスト"""
    results = []
    all_predictions = []

    df = df.copy()
    df['year_month'] = df['cdr_date'].dt.to_period('M')

    unique_months = sorted(df['year_month'].unique())

    start_idx = 3

    for month_idx in range(start_idx, min(len(unique_months) - 1, start_idx + n_months)):
        current_month = unique_months[month_idx]

        month_results = []
        test_month_data = df[df['year_month'] == current_month].copy()

        for idx, row in test_month_data.iterrows():
            target_date = row['cdr_date']

            # ホライズンの計算
            days_from_20th = (target_date - pd.Timestamp(target_date.year, target_date.month, 1)).days + 10
            horizon = max(5, int(days_from_20th * 5 / 7))

            df_features, feature_cols = create_features_for_prediction(df, horizon)

            # 訓練データ
            train_mask = (
                (df_features['year_month'] < unique_months[month_idx - 1]) |
                ((df_features['year_month'] == unique_months[month_idx - 1]) &
                 (df_features['cdr_date'].dt.day <= 20))
            )
            train_df = df_features[train_mask].dropna(subset=feature_cols + ['call_num'])

            if len(train_df) < 30:
                continue

            test_df = df_features[df_features['cdr_date'] == target_date].dropna(subset=feature_cols + ['call_num'])

            if len(test_df) == 0:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df['call_num']
            X_test = test_df[feature_cols]
            y_test = test_df['call_num']

            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            month_results.append({
                'date': target_date,
                'actual': y_test.values[0],
                'pred': y_pred[0]
            })

            all_predictions.append({
                'month': str(current_month),
                'date': target_date,
                'actual': y_test.values[0],
                'pred': y_pred[0]
            })

        if len(month_results) > 0:
            actuals = [r['actual'] for r in month_results]
            preds = [r['pred'] for r in month_results]

            mae = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))

            results.append({
                'month': str(current_month),
                'n_days': len(month_results),
                'mae': mae,
                'rmse': rmse,
                'actual_mean': np.mean(actuals),
                'pred_mean': np.mean(preds)
            })

    return pd.DataFrame(results), pd.DataFrame(all_predictions)


monthly_results, monthly_predictions = monthly_backtest(df_business, n_months=12)

print(f"\n[月次予測結果] ({len(monthly_results)}ヶ月分)")
print(f"  MAE平均:  {monthly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {monthly_results['rmse'].mean():.1f}件")
print(f"  実績平均: {monthly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {monthly_results['pred_mean'].mean():.1f}件")
print(f"  誤差率:   {monthly_results['mae'].mean() / monthly_results['actual_mean'].mean() * 100:.1f}%")

# =============================================================================
# 4. 特徴量重要度
# =============================================================================
print("\n" + "=" * 70)
print("特徴量重要度")
print("=" * 70)

df_features, feature_cols = create_features_for_prediction(df_business, forecast_horizon=5)
train_df = df_features.dropna(subset=feature_cols + ['call_num'])

X = train_df[feature_cols]
y = train_df['call_num']

model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n[特徴量重要度]")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:20s}: {row['importance']:.3f} {bar}")

# =============================================================================
# 5. 結果サマリー
# =============================================================================
print("\n" + "=" * 70)
print("結果サマリー（クリーニング済みデータ）")
print("=" * 70)

w_mae = weekly_results['mae'].mean()
w_err = weekly_results['mae'].mean() / weekly_results['actual_mean'].mean() * 100
m_mae = monthly_results['mae'].mean()
m_err = monthly_results['mae'].mean() / monthly_results['actual_mean'].mean() * 100

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│          コール件数予測 - クリーニング済みデータ版 結果              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【データクリーニング】                                               │
│   - call_num=0の日を休業日(closed_flag)として除外                    │
│   - 営業日のみで学習・予測（427日）                                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 【週次予測】水曜→翌週月〜金                                          │
│   MAE:    {w_mae:5.1f}件                                                  │
│   誤差率: {w_err:5.1f}%                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ 【月次予測】20日→翌月1ヶ月                                           │
│   MAE:    {m_mae:5.1f}件                                                  │
│   誤差率: {m_err:5.1f}%                                                    │
└─────────────────────────────────────────────────────────────────────┘
""")

# 結果保存
weekly_results.to_csv(f"{DATA_DIR}/weekly_backtest_cleaned.csv", index=False)
monthly_results.to_csv(f"{DATA_DIR}/monthly_backtest_cleaned.csv", index=False)
weekly_predictions.to_csv(f"{DATA_DIR}/weekly_predictions_cleaned.csv", index=False)
monthly_predictions.to_csv(f"{DATA_DIR}/monthly_predictions_cleaned.csv", index=False)

print("結果を保存しました:")
print(f"  - weekly_backtest_cleaned.csv")
print(f"  - monthly_backtest_cleaned.csv")
print(f"  - weekly_predictions_cleaned.csv")
print(f"  - monthly_predictions_cleaned.csv")
