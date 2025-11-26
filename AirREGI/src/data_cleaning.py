"""
データクリーニングモジュール

休日フラグの修正:
- 元のholiday_flagはカレンダー上の祝日
- 実際のコールセンター休業日はcall_num=0の日
- 新しいフラグ「closed_flag」を作成して実態を反映
"""

import pandas as pd
import numpy as np

BASE_DIR = "/home/user/gci_2025_winter/AirREGI"
PROCESSED_DATA_DIR = f"{BASE_DIR}/data/processed"


def load_and_clean_data(verbose=True):
    """
    データを読み込み、クリーニングを実施

    Returns:
    --------
    df : DataFrame
        クリーニング済みの全データ
    df_business : DataFrame
        営業日のみのデータ（予測対象）
    """
    if verbose:
        print("=" * 70)
        print("データクリーニング")
        print("=" * 70)

    # データ読み込み
    df = pd.read_csv(f"{PROCESSED_DATA_DIR}/merged_call_data_numeric.csv")
    df['cdr_date'] = pd.to_datetime(df['cdr_date'])
    df = df.sort_values('cdr_date').reset_index(drop=True)

    if verbose:
        print(f"\n元データ: {len(df)}日")

    # =================================================================
    # 1. 実際の休業日フラグを作成
    # =================================================================
    # call_num=0 の日を実際の休業日とする
    df['closed_flag'] = (df['call_num'] == 0).astype(int)

    if verbose:
        print(f"\n【休業日フラグの修正】")
        print(f"  元のholiday_flag=1: {df['holiday_flag'].sum()}日")
        print(f"  新しいclosed_flag=1: {df['closed_flag'].sum()}日")

        # 差分
        only_holiday = ((df['holiday_flag'] == 1) & (df['closed_flag'] == 0)).sum()
        only_closed = ((df['holiday_flag'] == 0) & (df['closed_flag'] == 1)).sum()
        print(f"  祝日だが営業: {only_holiday}日")
        print(f"  平日だが休業: {only_closed}日")

    # =================================================================
    # 2. 連続休業日数の計算（休日明け特徴量用）
    # =================================================================
    # 各日の直前に何日連続で休業だったかを計算
    consecutive_closed = []
    count = 0
    for i, row in df.iterrows():
        consecutive_closed.append(count)
        if row['closed_flag'] == 1:
            count += 1
        else:
            count = 0
    df['consecutive_closed_before'] = consecutive_closed

    if verbose:
        print(f"\n【連続休業日数（休日明け特徴量）】")
        # 営業日のみで集計
        business_mask = df['closed_flag'] == 0
        dist = df.loc[business_mask, 'consecutive_closed_before'].value_counts().sort_index()
        print(f"  分布（営業日のみ）:")
        for days, count in dist.head(10).items():
            print(f"    {days}日連休明け: {count}日")

    # =================================================================
    # 3. 営業日データの抽出
    # =================================================================
    df_business = df[df['closed_flag'] == 0].copy()
    df_business = df_business.reset_index(drop=True)

    if verbose:
        print(f"\n【営業日データ】")
        print(f"  営業日数: {len(df_business)}日")
        print(f"  期間: {df_business['cdr_date'].min().date()} ～ {df_business['cdr_date'].max().date()}")

        # 曜日分布
        dow_names = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金', 6: '土', 7: '日'}
        print(f"\n  曜日分布:")
        dow_dist = df_business['dow'].value_counts().sort_index()
        for dow, count in dow_dist.items():
            print(f"    {dow_names[dow]}曜日: {count}日")

    # =================================================================
    # 4. 異常値の確認
    # =================================================================
    if verbose:
        print(f"\n【営業日コール件数の統計】")
        stats = df_business['call_num'].describe()
        print(stats)

        # 外れ値の確認（平均±3σ）
        mean = df_business['call_num'].mean()
        std = df_business['call_num'].std()
        upper = mean + 3 * std
        lower = mean - 3 * std

        outliers = df_business[(df_business['call_num'] > upper) | (df_business['call_num'] < lower)]
        print(f"\n  外れ値（平均±3σ外）: {len(outliers)}日")
        if len(outliers) > 0:
            print(f"  閾値: {lower:.0f} ～ {upper:.0f}")
            for _, row in outliers.iterrows():
                print(f"    {row['cdr_date'].date()}: {row['call_num']}件")

    return df, df_business


def create_features_for_prediction(df, forecast_horizon):
    """
    予測用の特徴量を作成（リークなし）

    Parameters:
    -----------
    df : DataFrame
        営業日のみのデータ
    forecast_horizon : int
        予測ホライズン（何営業日先を予測するか）

    Returns:
    --------
    df : DataFrame
        特徴量追加済みデータ
    feature_cols : list
        使用する特徴量のカラム名リスト
    """
    df = df.copy()

    # カレンダー特徴量
    df['month'] = df['cdr_date'].dt.month
    df['day'] = df['cdr_date'].dt.day
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week.astype(int)

    # コール件数のラグ特徴量
    for lag in range(forecast_horizon, forecast_horizon + 5):
        df[f'call_lag_{lag}'] = df['call_num'].shift(lag)

    # コール件数の移動平均
    df['call_ma_5'] = df['call_num'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()
    df['call_ma_10'] = df['call_num'].shift(forecast_horizon).rolling(window=10, min_periods=1).mean()
    df['call_ma_20'] = df['call_num'].shift(forecast_horizon).rolling(window=20, min_periods=1).mean()

    # コール件数の標準偏差
    df['call_std_5'] = df['call_num'].shift(forecast_horizon).rolling(window=5, min_periods=1).std()

    # 前週同曜日
    df['call_same_dow_prev'] = df['call_num'].shift(forecast_horizon + 5)

    # アカウント取得数のラグ・移動平均
    df['acc_lag'] = df['acc_get_cnt'].shift(forecast_horizon)
    df['acc_ma_5'] = df['acc_get_cnt'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()
    df['acc_ma_10'] = df['acc_get_cnt'].shift(forecast_horizon).rolling(window=10, min_periods=1).mean()

    # Google Trends検索数のラグ・移動平均
    df['search_lag'] = df['search_cnt'].shift(forecast_horizon)
    df['search_ma_5'] = df['search_cnt'].shift(forecast_horizon).rolling(window=5, min_periods=1).mean()

    # 使用する特徴量カラム
    feature_cols = [
        'dow', 'month', 'day', 'week_of_year',
        'cm_flg', 'day_before_holiday_flag', 'consecutive_closed_before',
        f'call_lag_{forecast_horizon}', f'call_lag_{forecast_horizon+1}',
        f'call_lag_{forecast_horizon+2}', f'call_lag_{forecast_horizon+3}',
        f'call_lag_{forecast_horizon+4}',
        'call_ma_5', 'call_ma_10', 'call_ma_20',
        'call_std_5', 'call_same_dow_prev',
        'acc_lag', 'acc_ma_5', 'acc_ma_10',
        'search_lag', 'search_ma_5'
    ]

    # 存在するカラムのみ
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols


# =============================================================================
# メイン実行
# =============================================================================
if __name__ == "__main__":
    # データ読み込みとクリーニング
    df_all, df_business = load_and_clean_data(verbose=True)

    # クリーニング済みデータを保存
    output_path = f"{PROCESSED_DATA_DIR}/cleaned_data.csv"
    df_all.to_csv(output_path, index=False)
    print(f"\n【保存】")
    print(f"  クリーニング済みデータ: {output_path}")

    output_path_business = f"{PROCESSED_DATA_DIR}/business_days_data.csv"
    df_business.to_csv(output_path_business, index=False)
    print(f"  営業日データ: {output_path_business}")

    # 特徴量作成のテスト
    print("\n" + "=" * 70)
    print("特徴量作成テスト（ホライズン=5）")
    print("=" * 70)

    df_features, feature_cols = create_features_for_prediction(df_business, forecast_horizon=5)
    print(f"\n使用特徴量 ({len(feature_cols)}個):")
    for col in feature_cols:
        print(f"  - {col}")
