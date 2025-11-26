"""
コール件数予測システム - リークなし版 v2

データ取り扱いルール:
- キャンペーン(cm_flg): 計画があるので予測時点で既知 → 使用OK
- search_cnt, acc_get_cnt: 予測対象日の値は未知だが、予測実行日までの実績は使用OK
- ラグ特徴量: 予測実行日時点で確定している実績のみ使用

週次予測シナリオ:
- 予測実行日: 水曜（この日までの実績が確定）
- 予測対象: 翌週月〜金（5〜9営業日先）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "/home/user/gci_2025_winter/AirREGI"

print("=" * 70)
print("コール件数予測システム - リークなし版 v2")
print("=" * 70)

# =============================================================================
# 1. データ読み込み
# =============================================================================
print("\n" + "=" * 70)
print("1. データ読み込み")
print("=" * 70)

df = pd.read_csv(f"{DATA_DIR}/merged_call_data_numeric.csv")
df['cdr_date'] = pd.to_datetime(df['cdr_date'])
df = df.sort_values('cdr_date').reset_index(drop=True)

# 平日のみ（土日・祝日はコールセンター休業）
df_weekday = df[(df['holiday_flag'] == 0) & (df['dow'] <= 5)].copy()
df_weekday = df_weekday.reset_index(drop=True)

print(f"平日データ: {len(df_weekday)}日")
print(f"期間: {df_weekday['cdr_date'].min().date()} ～ {df_weekday['cdr_date'].max().date()}")

# =============================================================================
# 2. 使用可能な特徴量の定義
# =============================================================================
print("\n" + "=" * 70)
print("2. 使用可能な特徴量")
print("=" * 70)

print("""
【使用OK - 予測対象日のカレンダー情報】
- dow: 曜日
- month: 月
- day: 日
- week_of_year: 週番号
- day_before_holiday_flag: 休前日フラグ

【使用OK - 計画情報】
- cm_flg: キャンペーンフラグ（予測対象日の計画）

【使用OK - 予測実行日までの実績（ラグとして）】
- call_num: コール件数のラグ・移動平均
- acc_get_cnt: アカウント取得数のラグ・移動平均
- search_cnt: Google Trends検索数のラグ・移動平均
""")

# =============================================================================
# 3. リークなし特徴量の作成
# =============================================================================
print("\n" + "=" * 70)
print("3. リークなし特徴量の作成")
print("=" * 70)

def create_features_no_leak_v2(df, forecast_horizon):
    """
    リークなしの特徴量を作成 v2

    Parameters:
    -----------
    df : DataFrame
        平日のみのデータ（時系列順）
    forecast_horizon : int
        予測ホライズン（何営業日先を予測するか）

    Returns:
    --------
    DataFrame with features
    """
    df = df.copy()

    # カレンダー特徴量（予測対象日の情報なので使用OK）
    df['month'] = df['cdr_date'].dt.month
    df['day'] = df['cdr_date'].dt.day
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week.astype(int)

    # === コール件数のラグ特徴量 ===
    # 予測実行日時点で確定している実績
    for lag in range(forecast_horizon, forecast_horizon + 5):
        df[f'call_lag_{lag}'] = df['call_num'].shift(lag)

    # コール件数の移動平均（予測実行日基準）
    df['call_ma_5'] = df['call_num'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()
    df['call_ma_10'] = df['call_num'].shift(forecast_horizon).rolling(window=10, min_periods=1).mean()
    df['call_ma_20'] = df['call_num'].shift(forecast_horizon).rolling(window=20, min_periods=1).mean()

    # コール件数の標準偏差
    df['call_std_5'] = df['call_num'].shift(forecast_horizon).rolling(window=5, min_periods=1).std()

    # 前週同曜日
    df['call_same_dow_prev'] = df['call_num'].shift(forecast_horizon + 5)

    # === アカウント取得数のラグ特徴量 ===
    # 予測実行日までの実績を使用
    df['acc_lag'] = df['acc_get_cnt'].shift(forecast_horizon)
    df['acc_ma_5'] = df['acc_get_cnt'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()
    df['acc_ma_10'] = df['acc_get_cnt'].shift(forecast_horizon).rolling(window=10, min_periods=1).mean()

    # === Google Trends検索数のラグ特徴量 ===
    # 予測実行日までの実績を使用
    df['search_lag'] = df['search_cnt'].shift(forecast_horizon)
    df['search_ma_5'] = df['search_cnt'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()

    return df


# =============================================================================
# 4. 週次予測のバックテスト（リークなし v2）
# =============================================================================
print("\n" + "=" * 70)
print("4. 週次予測バックテスト（リークなし v2）")
print("=" * 70)
print("シナリオ: 水曜までの実績で翌週月〜金を予測")

def weekly_backtest_no_leak_v2(df, n_weeks=30):
    """
    週次予測のバックテスト（リークなし v2）
    """
    results = []
    all_predictions = []

    df = df.copy()
    df['year'] = df['cdr_date'].dt.year
    df['week'] = df['cdr_date'].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)

    unique_weeks = sorted(df['year_week'].unique())

    # 予測対象の各曜日に対応するホライズン
    horizons = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}

    start_idx = 10

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        week_results = []

        for target_dow, horizon in horizons.items():
            df_features = create_features_no_leak_v2(df, horizon)

            feature_cols = [
                'dow', 'month', 'day', 'week_of_year',
                'cm_flg', 'day_before_holiday_flag',
                f'call_lag_{horizon}', f'call_lag_{horizon+1}', f'call_lag_{horizon+2}',
                f'call_lag_{horizon+3}', f'call_lag_{horizon+4}',
                'call_ma_5', 'call_ma_10', 'call_ma_20',
                'call_std_5', 'call_same_dow_prev',
                'acc_lag', 'acc_ma_5', 'acc_ma_10',
                'search_lag', 'search_ma_5'
            ]

            feature_cols = [c for c in feature_cols if c in df_features.columns]

            train_mask = df_features['year_week'].isin(train_weeks)
            train_df = df_features[train_mask].dropna(subset=feature_cols + ['call_num'])

            if len(train_df) < 20:
                continue

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
                'actual': y_test.values[0],
                'pred': y_pred[0],
                'date': test_df['cdr_date'].values[0]
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


weekly_results, weekly_predictions = weekly_backtest_no_leak_v2(df_weekday, n_weeks=30)

print(f"\n[週次予測バックテスト結果] ({len(weekly_results)}週分)")
print(f"  MAE平均:  {weekly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {weekly_results['rmse'].mean():.1f}件")
print(f"  実績平均: {weekly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {weekly_results['pred_mean'].mean():.1f}件")
print(f"  誤差率:   {weekly_results['mae'].mean() / weekly_results['actual_mean'].mean() * 100:.1f}%")

# =============================================================================
# 5. 月次予測のバックテスト（リークなし v2）
# =============================================================================
print("\n" + "=" * 70)
print("5. 月次予測バックテスト（リークなし v2）")
print("=" * 70)
print("シナリオ: 20日までの実績で翌月1ヶ月分を予測")

def monthly_backtest_no_leak_v2(df, n_months=12):
    """
    月次予測のバックテスト（リークなし v2）
    """
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

            days_from_20th = (target_date - pd.Timestamp(target_date.year, target_date.month, 1)).days + 10
            horizon = max(5, int(days_from_20th * 5 / 7))

            df_features = create_features_no_leak_v2(df, horizon)

            feature_cols = [
                'dow', 'month', 'day', 'week_of_year',
                'cm_flg', 'day_before_holiday_flag',
                'call_ma_5', 'call_ma_10', 'call_ma_20',
                'call_std_5', 'call_same_dow_prev',
                'acc_lag', 'acc_ma_5', 'acc_ma_10',
                'search_lag', 'search_ma_5'
            ]

            for lag in range(horizon, min(horizon + 5, horizon + 10)):
                col = f'call_lag_{lag}'
                if col in df_features.columns:
                    feature_cols.append(col)

            feature_cols = [c for c in feature_cols if c in df_features.columns]

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


monthly_results, monthly_predictions = monthly_backtest_no_leak_v2(df_weekday, n_months=12)

print(f"\n[月次予測バックテスト結果] ({len(monthly_results)}ヶ月分)")
print(f"  MAE平均:  {monthly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {monthly_results['rmse'].mean():.1f}件")
print(f"  実績平均: {monthly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {monthly_results['pred_mean'].mean():.1f}件")
print(f"  誤差率:   {monthly_results['mae'].mean() / monthly_results['actual_mean'].mean() * 100:.1f}%")

# =============================================================================
# 6. 特徴量重要度（リークなし v2）
# =============================================================================
print("\n" + "=" * 70)
print("6. 特徴量重要度（リークなし v2）")
print("=" * 70)

horizon = 5
df_features = create_features_no_leak_v2(df_weekday, horizon)

feature_cols = [
    'dow', 'month', 'day', 'week_of_year',
    'cm_flg', 'day_before_holiday_flag',
    f'call_lag_{horizon}', f'call_lag_{horizon+1}', f'call_lag_{horizon+2}',
    f'call_lag_{horizon+3}', f'call_lag_{horizon+4}',
    'call_ma_5', 'call_ma_10', 'call_ma_20',
    'call_std_5', 'call_same_dow_prev',
    'acc_lag', 'acc_ma_5', 'acc_ma_10',
    'search_lag', 'search_ma_5'
]
feature_cols = [c for c in feature_cols if c in df_features.columns]

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
# 7. 結果サマリー
# =============================================================================
print("\n" + "=" * 70)
print("7. リークなし v2 結果サマリー")
print("=" * 70)

w_mae = weekly_results['mae'].mean()
w_err = weekly_results['mae'].mean() / weekly_results['actual_mean'].mean() * 100
m_mae = monthly_results['mae'].mean()
m_err = monthly_results['mae'].mean() / monthly_results['actual_mean'].mean() * 100

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│              コール件数予測 - リークなし v2 検証結果                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【使用特徴量】                                                       │
│   カレンダー: dow, month, day, week_of_year, day_before_holiday_flag│
│   計画情報:   cm_flg（予測対象日のキャンペーン計画）                  │
│   コール実績: ラグ(5-9日前), 移動平均(5,10,20日), 標準偏差, 前週同曜日│
│   外部実績:   acc_get_cnt, search_cnt のラグ・移動平均               │
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
weekly_results.to_csv(f"{DATA_DIR}/weekly_backtest_no_leak_v2.csv", index=False)
monthly_results.to_csv(f"{DATA_DIR}/monthly_backtest_no_leak_v2.csv", index=False)
weekly_predictions.to_csv(f"{DATA_DIR}/weekly_predictions_detail.csv", index=False)
monthly_predictions.to_csv(f"{DATA_DIR}/monthly_predictions_detail.csv", index=False)

print(f"結果を保存しました:")
print(f"  - weekly_backtest_no_leak_v2.csv")
print(f"  - monthly_backtest_no_leak_v2.csv")
print(f"  - weekly_predictions_detail.csv (日別詳細)")
print(f"  - monthly_predictions_detail.csv (日別詳細)")
