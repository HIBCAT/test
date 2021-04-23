import glob
import pandas as pd
import os
from collections import Counter
from functools import reduce


# April 23, 2021:
# Nike presentation code to create final datasets for multiple brands


def feature_2(brandwatch_path, clinecenter_path):
    # Brandwatch Dataset:

    # Part 1:
    col = ['publication_date', 'bing_liu_neg', 'bing_liu_pos']

    f2_clinecenter_01 = pd.read_csv(clinecenter_path, sep="\t", usecols=col)
    f2_clinecenter_01['publication_date'] = pd.to_datetime(f2_clinecenter_01['publication_date'])
    f2_clinecenter_01['publication_date_only'] = f2_clinecenter_01['publication_date'].dt.date
    f2_clinecenter_01['bing_liu_net_sentiment'] = [None] * len(f2_clinecenter_01)

    for i in range(len(f2_clinecenter_01)):
        if f2_clinecenter_01['bing_liu_pos'][i] != None:
            if f2_clinecenter_01['bing_liu_pos'][i] > f2_clinecenter_01['bing_liu_neg'][i]:
                f2_clinecenter_01['bing_liu_net_sentiment'][i] = 1
            elif f2_clinecenter_01['bing_liu_neg'][i] > f2_clinecenter_01['bing_liu_pos'][i]:
                f2_clinecenter_01['bing_liu_net_sentiment'][i] = -1
            else:
                f2_clinecenter_01['bing_liu_net_sentiment'][i] = 0
        else:
            pass

    f2_clinecenter_01.sort_values(by='publication_date_only', ascending=False, inplace=True)
    f2_clinecenter_01.reset_index(inplace=True, drop=True)

    # Now creating the main dataframe : clinecenter_02
    f2_clinecenter_02 = pd.DataFrame({"date": list(set(f2_clinecenter_01['publication_date_only']))})
    f2_clinecenter_02['date'] = pd.to_datetime(f2_clinecenter_02['date'])
    f2_clinecenter_02.sort_values(by='date', ascending=False, inplace=True)
    f2_clinecenter_02['cc_positive'] = 0
    f2_clinecenter_02['cc_negative'] = 0
    f2_clinecenter_02['cc_neutral'] = 0
    f2_clinecenter_02.reset_index(inplace=True, drop=True)

    # Now calculating the volume of the posts:
    # I will be ignoring the None type values in bing_liu_net_sentiment

    unique_index = pd.Index(f2_clinecenter_02['date'])

    for i in range(len(f2_clinecenter_01)):
        # Finding the matching index of dates in the main dataframes indexes
        index_match = unique_index.get_loc(f2_clinecenter_01['publication_date_only'][i])

        if f2_clinecenter_01['bing_liu_net_sentiment'][i] == 1:
            f2_clinecenter_02['cc_positive'][index_match] += 1
        elif f2_clinecenter_01['bing_liu_net_sentiment'][i] == 0:
            f2_clinecenter_02['cc_neutral'][index_match] += 1
        elif f2_clinecenter_01['bing_liu_net_sentiment'][i] < 0:
            f2_clinecenter_02['cc_negative'][index_match] += 1
        else:
            pass

    f2_clinecenter_02['cc_volume'] = f2_clinecenter_02['cc_negative'] + f2_clinecenter_02['cc_neutral'] + \
                                     f2_clinecenter_02['cc_positive']

    # Part 2: Preparing BrandWatch Dataset

    f2_brandwatch_01 = pd.read_csv(brandwatch_path)
    f2_brandwatch_01.drop(columns=['net_sentiment'], inplace=True)
    f2_brandwatch_01['days'] = pd.to_datetime(f2_brandwatch_01['days'])
    f2_brandwatch_01.sort_values(by='days', ascending=False, inplace=True)
    f2_brandwatch_01.reset_index(inplace=True, drop=True)
    f2_brandwatch_01.rename(columns={'days': 'date',
                                     'positive': 'bw_positive',
                                     'neutral': 'bw_neutral',
                                     'negative': 'bw_negative',
                                     'volume': 'bw_volume'}, inplace=True)

    # Part 3: Appending the datasets in part 1 and part 2:
    f2_cc_bw = pd.merge(f2_brandwatch_01, f2_clinecenter_02, on="date", how="inner")

    # Now filling the 3 month's moving average of all the columns

    flag = len(f2_cc_bw) - 89

    bw_3ma_positive = [None] * len(f2_cc_bw)
    bw_3ma_negative = [None] * len(f2_cc_bw)
    bw_3ma_neutral = [None] * len(f2_cc_bw)
    bw_3ma_volume = [None] * len(f2_cc_bw)

    cc_3ma_positive = [None] * len(f2_cc_bw)
    cc_3ma_negative = [None] * len(f2_cc_bw)
    cc_3ma_neutral = [None] * len(f2_cc_bw)
    cc_3ma_volume = [None] * len(f2_cc_bw)

    for i in range(len(f2_cc_bw)):
        if i != flag:

            bw_3ma_positive[i] = (sum(f2_cc_bw['bw_positive'][i:i + 90]) / 90)
            bw_3ma_negative[i] = (sum(f2_cc_bw['bw_negative'][i:i + 90]) / 90)
            bw_3ma_neutral[i] = (sum(f2_cc_bw['bw_neutral'][i:i + 90]) / 90)
            bw_3ma_volume[i] = (sum(f2_cc_bw['bw_volume'][i:i + 90]) / 90)

            cc_3ma_positive[i] = (sum(f2_cc_bw['cc_positive'][i:i + 90]) / 90)
            cc_3ma_negative[i] = (sum(f2_cc_bw['cc_negative'][i:i + 90]) / 90)
            cc_3ma_neutral[i] = (sum(f2_cc_bw['cc_neutral'][i:i + 90]) / 90)
            cc_3ma_volume[i] = (sum(f2_cc_bw['cc_volume'][i:i + 90]) / 90)

        else:
            break

    f2_cc_bw['bw_3ma_positive'] = bw_3ma_positive
    f2_cc_bw['bw_3ma_negative'] = bw_3ma_negative
    f2_cc_bw['bw_3ma_neutral'] = bw_3ma_neutral
    f2_cc_bw['bw_3ma_volume'] = bw_3ma_volume

    f2_cc_bw['cc_3ma_positive'] = cc_3ma_positive
    f2_cc_bw['cc_3ma_negative'] = cc_3ma_negative
    f2_cc_bw['cc_3ma_neutral'] = cc_3ma_neutral
    f2_cc_bw['cc_3ma_volume'] = cc_3ma_volume

    # Now normalizing all the columns on a scale of 0 to 100.
    # Then standardizing all the normalized values.

    f2_cc_bw['bw_std_positive'] = [None] * len(f2_cc_bw)
    f2_cc_bw['bw_std_negative'] = [None] * len(f2_cc_bw)
    f2_cc_bw['bw_std_neutral'] = [None] * len(f2_cc_bw)
    f2_cc_bw['bw_std_volume'] = [None] * len(f2_cc_bw)

    f2_cc_bw['cc_std_positive'] = [None] * len(f2_cc_bw)
    f2_cc_bw['cc_std_negative'] = [None] * len(f2_cc_bw)
    f2_cc_bw['cc_std_neutral'] = [None] * len(f2_cc_bw)
    f2_cc_bw['cc_std_volume'] = [None] * len(f2_cc_bw)

    columns = ['bw_positive', 'bw_neutral', 'bw_negative', 'bw_volume',
               'cc_positive', 'cc_negative', 'cc_neutral', 'cc_volume',

               'bw_3ma_positive', 'bw_3ma_neutral', 'bw_3ma_negative', 'bw_3ma_volume',
               'cc_3ma_positive', 'cc_3ma_negative', 'cc_3ma_neutral', 'cc_3ma_volume',
               ]

    # Creating columns on the go
    for i in columns:
        # 1. Standardized volume
        # x : value to be standardized
        # bw_vol_mean = mean of the range
        # bw_vol_std = standard deviation of the range

        f2_cc_bw_mean = f2_cc_bw[i].mean()
        f2_cc_bw_std = f2_cc_bw[i].std()
        f2_cc_bw[f'{i}_std_0_100'] = (f2_cc_bw[i] - f2_cc_bw_mean) / f2_cc_bw_std

        # 2. Normalizing the columns on a scale of 0 to 100.
        # Normalization: (b-a) * [(x-y)/(z-y)] + a
        # (a,b): Range of normalized score
        # (0, 100)
        # x : Value to be normalized
        # y : Min value from the range
        # z : Max value from the range

        a = 0
        b = 100
        y = f2_cc_bw[f'{i}_std_0_100'].min()
        z = f2_cc_bw[f'{i}_std_0_100'].max()
        f2_cc_bw[f'{i}_std_0_100'] = (b - a) * ((f2_cc_bw[f'{i}_std_0_100'] - y) / (z - y)) + a
        # For loop ends over here. Beware

    # Now melt all the columns to get the final dataset for the feature 2
    f2_cc_bw = pd.melt(f2_cc_bw, id_vars=["date"],
                       value_vars=['bw_positive', 'bw_neutral', 'bw_negative', 'bw_volume',
                                   'cc_positive', 'cc_negative', 'cc_neutral', 'cc_volume',
                                   'bw_3ma_positive', 'bw_3ma_negative', 'bw_3ma_neutral', 'bw_3ma_volume',
                                   'cc_3ma_positive', 'cc_3ma_negative', 'cc_3ma_neutral', 'cc_3ma_volume',
                                   'bw_std_positive', 'bw_std_negative', 'bw_std_neutral', 'bw_std_volume',
                                   'cc_std_positive', 'cc_std_negative', 'cc_std_neutral', 'cc_std_volume',
                                   'bw_positive_std_0_100', 'bw_neutral_std_0_100',
                                   'bw_negative_std_0_100', 'bw_volume_std_0_100', 'cc_positive_std_0_100',
                                   'cc_negative_std_0_100', 'cc_neutral_std_0_100', 'cc_volume_std_0_100',
                                   'bw_3ma_positive_std_0_100', 'bw_3ma_neutral_std_0_100',
                                   'bw_3ma_negative_std_0_100', 'bw_3ma_volume_std_0_100',
                                   'cc_3ma_positive_std_0_100', 'cc_3ma_negative_std_0_100',
                                   'cc_3ma_neutral_std_0_100', 'cc_3ma_volume_std_0_100'], var_name="attributes",
                       value_name="values")
    f2_cc_bw.dropna(inplace=True)
    return f2_cc_bw


company_names = ['brinks', 'CocaCola', 'KPMG',
                'Microsoft', 'MLB', 'Monday_com',
                'Nike', 'Starbucks', 'UnitedAirline']

for filename in company_names:
    clinecenter_path = f"Friday_Assignment/ClineCenter/{filename}.tsv"
    brandwatch_path = f"Friday_Assignment/BrandWatch/BW_Final/{filename}.csv"
    feature_2(brandwatch_path, clinecenter_path).to_csv(f'Friday_Assignment/Feature_2_Datasets/{filename}.csv', index=False)

