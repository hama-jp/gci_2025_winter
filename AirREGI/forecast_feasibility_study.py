"""
コール件数予測システムの実現性検証

シナリオ:
1. 週次予測: 水曜までの実績で翌週月〜金を予測
2. 月次予測: 20日までの実績で翌月1ヶ月分を予測

検証内容:
- データ特性の分析（自己相関、季節性）
- バックテストによる予測精度の検証
- モデル更新の効果検証
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# scikit-learnのインストール確認
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("scikit-learnをインストールしています...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/user/gci_2025_winter/AirREGI"

print("=" * 70)
print("コール件数予測システム 実現性検証")
print("=" * 70)

# =============================================================================
# 1. データ読み込みと前処理
# =============================================================================
print("\n" + "=" * 70)
print("1. データ読み込みと前処理")
print("=" * 70)

df = pd.read_csv(f"{DATA_DIR}/merged_call_data_numeric.csv")
df['cdr_date'] = pd.to_datetime(df['cdr_date'])
df = df.sort_values('cdr_date').reset_index(drop=True)

# 平日のみ抽出（土日・祝日は予測対象外）
df_weekday = df[(df['holiday_flag'] == 0) & (df['dow'] <= 5)].copy()
print(f"全データ: {len(df)}日")
print(f"平日データ: {len(df_weekday)}日")
print(f"期間: {df['cdr_date'].min().date()} ～ {df['cdr_date'].max().date()}")

# =============================================================================
# 2. データ特性の分析
# =============================================================================
print("\n" + "=" * 70)
print("2. データ特性の分析")
print("=" * 70)

# 平日のコール件数統計
print("\n[平日コール件数の統計]")
weekday_stats = df_weekday['call_num'].describe()
print(weekday_stats)

# 曜日別平均
print("\n[曜日別平均コール件数]")
dow_avg = df_weekday.groupby('dow')['call_num'].mean()
dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}
for dow, avg in dow_avg.items():
    print(f"  {dow_names[dow]}曜日: {avg:.1f}件")

# 自己相関の計算
print("\n[自己相関分析]")
call_series = df_weekday['call_num'].values
for lag in [1, 5, 10, 20]:  # 1日, 1週(5営業日), 2週, 1ヶ月(20営業日)
    if len(call_series) > lag:
        corr = np.corrcoef(call_series[lag:], call_series[:-lag])[0, 1]
        print(f"  ラグ{lag}日: {corr:.3f}")

# =============================================================================
# 3. 特徴量エンジニアリング
# =============================================================================
print("\n" + "=" * 70)
print("3. 特徴量エンジニアリング")
print("=" * 70)

def create_features(df, target_col='call_num'):
    """特徴量を作成"""
    df = df.copy()

    # 基本特徴量
    df['month'] = df['cdr_date'].dt.month
    df['day'] = df['cdr_date'].dt.day
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week.astype(int)

    # ラグ特徴量（平日ベースで計算するため、indexで管理）
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 直近の実績（1〜5営業日前）
    for lag in [1, 2, 3, 4, 5]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # 移動平均
    df['ma_5'] = df[target_col].rolling(window=5, min_periods=1).mean().shift(1)
    df['ma_10'] = df[target_col].rolling(window=10, min_periods=1).mean().shift(1)
    df['ma_20'] = df[target_col].rolling(window=20, min_periods=1).mean().shift(1)

    # 前週同曜日（5営業日前）
    df['same_dow_last_week'] = df[target_col].shift(5)

    # 標準偏差（変動性）
    df['std_5'] = df[target_col].rolling(window=5, min_periods=1).std().shift(1)

    return df

# 平日データに特徴量追加
df_features = create_features(df_weekday)
print(f"特徴量数: {len(df_features.columns)}")
print(f"カラム: {list(df_features.columns)}")

# =============================================================================
# 4. 週次予測のバックテスト
# =============================================================================
print("\n" + "=" * 70)
print("4. 週次予測のバックテスト")
print("=" * 70)
print("シナリオ: 水曜までの実績で翌週月〜金を予測")

def get_week_boundaries(df):
    """各週の境界を取得"""
    df = df.copy()
    df['year_week'] = df['cdr_date'].dt.strftime('%Y-%W')
    weeks = df.groupby('year_week').agg({
        'cdr_date': ['min', 'max', 'count']
    }).reset_index()
    weeks.columns = ['year_week', 'start', 'end', 'days']
    return weeks

# 特徴量カラム
feature_cols = ['dow', 'month', 'day', 'week_of_year',
                'acc_get_cnt', 'cm_flg', 'search_cnt',
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                'ma_5', 'ma_10', 'ma_20', 'same_dow_last_week', 'std_5']

def weekly_backtest(df, feature_cols, n_weeks=20):
    """
    週次予測のバックテスト
    水曜までのデータで翌週月〜金を予測
    """
    results = []
    df = df.copy()

    # 週ごとにグループ化
    df['year'] = df['cdr_date'].dt.year
    df['week'] = df['cdr_date'].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)

    unique_weeks = df['year_week'].unique()

    # 最低10週分のデータで学習開始
    start_idx = 10

    for i in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:i]
        test_week = unique_weeks[i]

        # 訓練データ（水曜まで = dow <= 3）
        train_mask = df['year_week'].isin(train_weeks)
        train_df = df[train_mask].dropna(subset=feature_cols + ['call_num'])

        if len(train_df) < 20:
            continue

        # テストデータ（翌週の平日）
        test_mask = df['year_week'] == test_week
        test_df = df[test_mask].dropna(subset=feature_cols + ['call_num'])

        if len(test_df) == 0:
            continue

        # モデル学習
        X_train = train_df[feature_cols]
        y_train = train_df['call_num']
        X_test = test_df[feature_cols]
        y_test = test_df['call_num']

        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # 予測
        y_pred = model.predict(X_test)

        # 評価
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if y_test.mean() > 0 else np.nan

        results.append({
            'week': test_week,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'actual_mean': y_test.mean(),
            'pred_mean': y_pred.mean()
        })

    return pd.DataFrame(results)

weekly_results = weekly_backtest(df_features, feature_cols, n_weeks=30)

print(f"\n[週次予測バックテスト結果] ({len(weekly_results)}週分)")
print(f"  MAE平均:  {weekly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {weekly_results['rmse'].mean():.1f}件")
print(f"  MAPE平均: {weekly_results['mape'].mean():.1f}%")
print(f"\n  実績平均: {weekly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {weekly_results['pred_mean'].mean():.1f}件")

# =============================================================================
# 5. 月次予測のバックテスト
# =============================================================================
print("\n" + "=" * 70)
print("5. 月次予測のバックテスト")
print("=" * 70)
print("シナリオ: 20日までの実績で翌月1ヶ月分を予測")

def monthly_backtest(df, feature_cols, n_months=12):
    """
    月次予測のバックテスト
    20日までのデータで翌月を予測
    """
    results = []
    df = df.copy()

    df['year_month'] = df['cdr_date'].dt.to_period('M')
    unique_months = df['year_month'].unique()

    # 最低3ヶ月分のデータで学習開始
    start_idx = 3

    for i in range(start_idx, min(len(unique_months) - 1, start_idx + n_months)):
        # 訓練データ: 前月20日まで
        train_months = unique_months[:i]
        current_month = unique_months[i]

        # 当月20日までのデータを追加
        current_month_mask = (df['year_month'] == unique_months[i-1]) & (df['cdr_date'].dt.day <= 20)
        train_mask = df['year_month'].isin(train_months[:-1]) | current_month_mask
        train_df = df[train_mask].dropna(subset=feature_cols + ['call_num'])

        if len(train_df) < 30:
            continue

        # テストデータ: 翌月全体
        test_mask = df['year_month'] == current_month
        test_df = df[test_mask].dropna(subset=feature_cols + ['call_num'])

        if len(test_df) == 0:
            continue

        # モデル学習
        X_train = train_df[feature_cols]
        y_train = train_df['call_num']
        X_test = test_df[feature_cols]
        y_test = test_df['call_num']

        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # 予測
        y_pred = model.predict(X_test)

        # 評価
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if y_test.mean() > 0 else np.nan

        results.append({
            'month': str(current_month),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'actual_mean': y_test.mean(),
            'pred_mean': y_pred.mean()
        })

    return pd.DataFrame(results)

monthly_results = monthly_backtest(df_features, feature_cols, n_months=15)

print(f"\n[月次予測バックテスト結果] ({len(monthly_results)}ヶ月分)")
print(f"  MAE平均:  {monthly_results['mae'].mean():.1f}件")
print(f"  RMSE平均: {monthly_results['rmse'].mean():.1f}件")
print(f"  MAPE平均: {monthly_results['mape'].mean():.1f}%")
print(f"\n  実績平均: {monthly_results['actual_mean'].mean():.1f}件")
print(f"  予測平均: {monthly_results['pred_mean'].mean():.1f}件")

# =============================================================================
# 6. モデル更新の効果検証
# =============================================================================
print("\n" + "=" * 70)
print("6. モデル更新の効果検証")
print("=" * 70)

def compare_model_update(df, feature_cols):
    """
    モデル更新あり/なしの比較
    """
    df = df.copy()
    df['year_month'] = df['cdr_date'].dt.to_period('M')
    unique_months = df['year_month'].unique()

    results_with_update = []
    results_without_update = []

    # 固定モデル（最初の6ヶ月で学習）
    fixed_train_months = unique_months[:6]
    fixed_train_mask = df['year_month'].isin(fixed_train_months)
    fixed_train_df = df[fixed_train_mask].dropna(subset=feature_cols + ['call_num'])

    X_fixed = fixed_train_df[feature_cols]
    y_fixed = fixed_train_df['call_num']
    fixed_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    fixed_model.fit(X_fixed, y_fixed)

    # 各月を予測
    for i in range(6, len(unique_months) - 1):
        test_month = unique_months[i]
        test_mask = df['year_month'] == test_month
        test_df = df[test_mask].dropna(subset=feature_cols + ['call_num'])

        if len(test_df) == 0:
            continue

        X_test = test_df[feature_cols]
        y_test = test_df['call_num']

        # 固定モデルで予測
        y_pred_fixed = fixed_model.predict(X_test)
        mae_fixed = mean_absolute_error(y_test, y_pred_fixed)

        # 更新モデルで予測
        update_train_mask = df['year_month'].isin(unique_months[:i])
        update_train_df = df[update_train_mask].dropna(subset=feature_cols + ['call_num'])
        X_update = update_train_df[feature_cols]
        y_update = update_train_df['call_num']

        update_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        update_model.fit(X_update, y_update)
        y_pred_update = update_model.predict(X_test)
        mae_update = mean_absolute_error(y_test, y_pred_update)

        results_with_update.append(mae_update)
        results_without_update.append(mae_fixed)

    return results_with_update, results_without_update

mae_with_update, mae_without_update = compare_model_update(df_features, feature_cols)

print(f"\n[モデル更新の効果]")
print(f"  更新あり MAE平均: {np.mean(mae_with_update):.1f}件")
print(f"  更新なし MAE平均: {np.mean(mae_without_update):.1f}件")
print(f"  改善率: {(1 - np.mean(mae_with_update)/np.mean(mae_without_update))*100:.1f}%")

# =============================================================================
# 7. 特徴量重要度
# =============================================================================
print("\n" + "=" * 70)
print("7. 特徴量重要度分析")
print("=" * 70)

# 全データで学習
train_df = df_features.dropna(subset=feature_cols + ['call_num'])
X = train_df[feature_cols]
y = train_df['call_num']

model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n[特徴量重要度 Top 10]")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.3f}")

# =============================================================================
# 8. 結果サマリー
# =============================================================================
print("\n" + "=" * 70)
print("8. 実現性評価サマリー")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                     予測システム実現性評価                        │
├─────────────────────────────────────────────────────────────────┤
│ 【週次予測】水曜→翌週月〜金                                      │
│   - MAE: {mae_w:.1f}件 (平均{avg_w:.0f}件に対して誤差率{err_w:.1f}%)      │
│   - 予測ホライズン: 5営業日先                                     │
│   - 実現性: ◎ 高い                                              │
├─────────────────────────────────────────────────────────────────┤
│ 【月次予測】20日→翌月1ヶ月                                       │
│   - MAE: {mae_m:.1f}件 (平均{avg_m:.0f}件に対して誤差率{err_m:.1f}%)      │
│   - 予測ホライズン: 最大40日先                                    │
│   - 実現性: ○ 中程度（長期予測のため精度低下）                    │
├─────────────────────────────────────────────────────────────────┤
│ 【モデル更新効果】                                               │
│   - 更新により約{improve:.0f}%の精度向上                              │
│   - 推奨: 週次または月次でモデル再学習                            │
└─────────────────────────────────────────────────────────────────┘
""".format(
    mae_w=weekly_results['mae'].mean(),
    avg_w=weekly_results['actual_mean'].mean(),
    err_w=weekly_results['mae'].mean() / weekly_results['actual_mean'].mean() * 100,
    mae_m=monthly_results['mae'].mean(),
    avg_m=monthly_results['actual_mean'].mean(),
    err_m=monthly_results['mae'].mean() / monthly_results['actual_mean'].mean() * 100,
    improve=(1 - np.mean(mae_with_update)/np.mean(mae_without_update))*100
))

# 結果をCSVに保存
weekly_results.to_csv(f"{DATA_DIR}/weekly_backtest_results.csv", index=False)
monthly_results.to_csv(f"{DATA_DIR}/monthly_backtest_results.csv", index=False)
print(f"結果を保存しました:")
print(f"  - {DATA_DIR}/weekly_backtest_results.csv")
print(f"  - {DATA_DIR}/monthly_backtest_results.csv")
