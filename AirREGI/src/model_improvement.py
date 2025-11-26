"""
モデル改善実験

ベースライン（前週同曜日）を上回るモデルを探索
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_cleaning import load_and_clean_data

BASE_DIR = "/home/user/gci_2025_winter/AirREGI"

print("=" * 70)
print("モデル改善実験: ベースラインを上回るモデルを探索")
print("=" * 70)

# データ読み込み
df_all, df_business = load_and_clean_data(verbose=False)

df = df_business.copy()
df['year'] = df['cdr_date'].dt.year
df['week'] = df['cdr_date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
df['month'] = df['cdr_date'].dt.month

# =============================================================================
# 特徴量作成（曜日別にベースラインと同じラグを使用）
# =============================================================================
# ベースラインと同じラグ構造
# 月火水: 5営業日前（前週同曜日）
# 木金: 10営業日前（前々週同曜日）
lag_by_dow = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10}

for dow in [1, 2, 3, 4, 5]:
    lag = lag_by_dow[dow]
    # ベースライン特徴量（前週/前々週同曜日）
    df[f'baseline_lag_{dow}'] = df['call_num'].shift(lag)
    # 追加ラグ
    df[f'lag_{dow}_plus1'] = df['call_num'].shift(lag + 1)
    df[f'lag_{dow}_plus2'] = df['call_num'].shift(lag + 2)
    # 移動平均（ラグ以降のデータのみ使用）
    df[f'ma5_{dow}'] = df['call_num'].shift(lag).rolling(5, min_periods=1).mean()
    df[f'ma10_{dow}'] = df['call_num'].shift(lag).rolling(10, min_periods=1).mean()

# 曜日共通の特徴量
df['acc_lag_5'] = df['acc_get_cnt'].shift(5)
df['acc_ma_5'] = df['acc_get_cnt'].shift(5).rolling(5, min_periods=1).mean()
df['search_lag_5'] = df['search_cnt'].shift(5)

unique_weeks = sorted(df['year_week'].unique())
start_idx = 10
n_weeks = 30


def evaluate_model(model_class, model_params, feature_sets, name):
    """モデルを評価"""
    results = []

    for week_idx in range(start_idx, min(len(unique_weeks) - 1, start_idx + n_weeks)):
        train_weeks = unique_weeks[:week_idx]
        test_week = unique_weeks[week_idx]

        week_actuals = []
        week_preds = []
        week_baselines = []

        for dow in [1, 2, 3, 4, 5]:
            # 曜日固有の特徴量を選択
            features = feature_sets[dow]

            # 訓練データ
            train_mask = df['year_week'].isin(train_weeks) & (df['dow'] == dow)
            train_df = df[train_mask].dropna(subset=features + ['call_num'])

            if len(train_df) < 10:
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

            model = model_class(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            week_actuals.append(y_test.values[0])
            week_preds.append(y_pred[0])
            week_baselines.append(test_df[f'baseline_lag_{dow}'].values[0])

        if len(week_actuals) > 0:
            mae = mean_absolute_error(week_actuals, week_preds)
            baseline_mae = mean_absolute_error(week_actuals, week_baselines)
            results.append({
                'week': test_week,
                'mae': mae,
                'baseline_mae': baseline_mae,
                'actual_mean': np.mean(week_actuals)
            })

    result_df = pd.DataFrame(results)
    avg_mae = result_df['mae'].mean()
    avg_baseline = result_df['baseline_mae'].mean()
    avg_actual = result_df['actual_mean'].mean()

    return {
        'name': name,
        'mae': avg_mae,
        'baseline_mae': avg_baseline,
        'error_rate': avg_mae / avg_actual * 100,
        'improvement': (avg_baseline - avg_mae) / avg_baseline * 100
    }


print("\n" + "=" * 70)
print("実験1: ベースラインのみ（前週/前々週同曜日）")
print("=" * 70)

# ベースラインのみの特徴量セット
baseline_features = {
    dow: [f'baseline_lag_{dow}'] for dow in [1, 2, 3, 4, 5]
}

result = evaluate_model(Ridge, {'alpha': 1.0}, baseline_features, "Ridge（ベースラインのみ）")
print(f"  {result['name']}: MAE={result['mae']:.1f}, 誤差率={result['error_rate']:.1f}%, 改善率={result['improvement']:+.1f}%")


print("\n" + "=" * 70)
print("実験2: ベースライン + 曜日情報")
print("=" * 70)

# ベースライン + 曜日
features_with_dow = {
    dow: [f'baseline_lag_{dow}', 'dow', 'month'] for dow in [1, 2, 3, 4, 5]
}

for model_class, params, name in [
    (Ridge, {'alpha': 1.0}, "Ridge"),
    (Ridge, {'alpha': 10.0}, "Ridge(alpha=10)"),
    (Lasso, {'alpha': 1.0}, "Lasso"),
]:
    result = evaluate_model(model_class, params, features_with_dow, name)
    print(f"  {result['name']}: MAE={result['mae']:.1f}, 誤差率={result['error_rate']:.1f}%, 改善率={result['improvement']:+.1f}%")


print("\n" + "=" * 70)
print("実験3: ベースライン + 移動平均")
print("=" * 70)

# ベースライン + 移動平均
features_with_ma = {
    dow: [f'baseline_lag_{dow}', f'ma5_{dow}', f'ma10_{dow}'] for dow in [1, 2, 3, 4, 5]
}

for model_class, params, name in [
    (Ridge, {'alpha': 1.0}, "Ridge"),
    (Ridge, {'alpha': 10.0}, "Ridge(alpha=10)"),
    (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}, "GB(shallow)"),
]:
    result = evaluate_model(model_class, params, features_with_ma, name)
    print(f"  {result['name']}: MAE={result['mae']:.1f}, 誤差率={result['error_rate']:.1f}%, 改善率={result['improvement']:+.1f}%")


print("\n" + "=" * 70)
print("実験4: ベースライン + 外部変数（アカウント取得数）")
print("=" * 70)

# ベースライン + アカウント取得数
features_with_acc = {
    dow: [f'baseline_lag_{dow}', 'acc_lag_5', 'acc_ma_5'] for dow in [1, 2, 3, 4, 5]
}

for model_class, params, name in [
    (Ridge, {'alpha': 1.0}, "Ridge"),
    (Ridge, {'alpha': 10.0}, "Ridge(alpha=10)"),
    (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}, "GB(shallow)"),
]:
    result = evaluate_model(model_class, params, features_with_acc, name)
    print(f"  {result['name']}: MAE={result['mae']:.1f}, 誤差率={result['error_rate']:.1f}%, 改善率={result['improvement']:+.1f}%")


print("\n" + "=" * 70)
print("実験5: ベースライン + 複数ラグ + 移動平均 + 外部変数")
print("=" * 70)

# フル特徴量
full_features = {
    dow: [
        f'baseline_lag_{dow}', f'lag_{dow}_plus1', f'lag_{dow}_plus2',
        f'ma5_{dow}', f'ma10_{dow}',
        'acc_lag_5', 'acc_ma_5', 'search_lag_5',
        'month', 'cm_flg', 'day_before_holiday_flag'
    ] for dow in [1, 2, 3, 4, 5]
}

for model_class, params, name in [
    (Ridge, {'alpha': 1.0}, "Ridge"),
    (Ridge, {'alpha': 10.0}, "Ridge(alpha=10)"),
    (Ridge, {'alpha': 100.0}, "Ridge(alpha=100)"),
    (ElasticNet, {'alpha': 1.0, 'l1_ratio': 0.5}, "ElasticNet"),
    (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 2, 'random_state': 42}, "GB(d=2)"),
    (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}, "GB(d=3)"),
]:
    result = evaluate_model(model_class, params, full_features, name)
    print(f"  {result['name']}: MAE={result['mae']:.1f}, 誤差率={result['error_rate']:.1f}%, 改善率={result['improvement']:+.1f}%")


print("\n" + "=" * 70)
print("実験6: 曜日別モデル（各曜日で最適化）")
print("=" * 70)

# 曜日別に最適な特徴量を使用
# 月火水は前週同曜日が有効、木金は前々週なので追加情報が有効かも

results_by_dow = []
for dow in [1, 2, 3, 4, 5]:
    dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}

    # その曜日のデータのみ使用
    train_weeks = unique_weeks[5:start_idx]
    test_weeks = unique_weeks[start_idx:start_idx + n_weeks]

    features = [f'baseline_lag_{dow}', f'ma5_{dow}', 'acc_lag_5']

    actuals = []
    preds = []
    baselines = []

    for test_week in test_weeks:
        train_mask = df['year_week'].isin(train_weeks) & (df['dow'] == dow)
        train_df = df[train_mask].dropna(subset=features + ['call_num'])

        test_mask = (df['year_week'] == test_week) & (df['dow'] == dow)
        test_df = df[test_mask].dropna(subset=features + ['call_num'])

        if len(train_df) < 5 or len(test_df) == 0:
            continue

        X_train = train_df[features]
        y_train = train_df['call_num']
        X_test = test_df[features]
        y_test = test_df['call_num']

        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        actuals.append(y_test.values[0])
        preds.append(y_pred[0])
        baselines.append(test_df[f'baseline_lag_{dow}'].values[0])

        # 訓練データを更新（拡張ウィンドウ）
        train_weeks = unique_weeks[5:unique_weeks.index(test_week)]

    if len(actuals) > 0:
        mae = mean_absolute_error(actuals, preds)
        baseline_mae = mean_absolute_error(actuals, baselines)
        improvement = (baseline_mae - mae) / baseline_mae * 100
        results_by_dow.append({
            'dow': dow,
            'dow_name': dow_names[dow],
            'mae': mae,
            'baseline_mae': baseline_mae,
            'improvement': improvement
        })
        print(f"  {dow_names[dow]}曜日: MAE={mae:.1f} vs ベースライン={baseline_mae:.1f}, 改善率={improvement:+.1f}%")

# 全体
if results_by_dow:
    total_mae = np.mean([r['mae'] for r in results_by_dow])
    total_baseline = np.mean([r['baseline_mae'] for r in results_by_dow])
    total_improvement = (total_baseline - total_mae) / total_baseline * 100
    print(f"\n  全体: MAE={total_mae:.1f} vs ベースライン={total_baseline:.1f}, 改善率={total_improvement:+.1f}%")


print("\n" + "=" * 70)
print("サマリー")
print("=" * 70)
print("""
目標: ベースライン（前週同曜日）MAE ≈ 42件 を上回る

改善のポイント:
1. 曜日別のラグ構造をベースラインと合わせる
2. 特徴量を絞って過学習を防ぐ
3. 正則化の強いモデル（Ridge, Lasso）を使用
4. 木の深さを浅くして汎化性能を上げる
""")
