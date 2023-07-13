import scipy

import pandas as pd


def read_in_csv():
    data = pd.read_csv("../data/data.csv")
    data.rename(columns={0: 'Date'}, inplace=False)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[1:-1]
    data = data[cols]
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data.index = data.index - pd.Timedelta(days=1)

    return data


def consecutive_nans(row):
    max_consec = 0
    current_consec = 0
    location_of_nans = {}

    for key, value in row.items():
        if pd.isna(value):

            if current_consec == 0:
                start_date = key
                location_of_nans[key] = pd.Timedelta(1, 'd')
            else:
                location_of_nans[start_date] += pd.Timedelta(1, 'd')
            current_consec += 1
        else:
            current_consec = 0

    return location_of_nans


def calculate_missing_values(df, to_fill, missing_data):
    data_nans_removed = df
    for station in to_fill.index:
        usable_indices = to_fill.loc[station, 'usable']
        x_index = to_fill.loc[station, 'x']
        x = df.loc[usable_indices, x_index]
        y = df.loc[usable_indices, station]
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)

        for start_date, consecutive_day in missing_data.loc[station].items():
            end_date = start_date + consecutive_day
            day = start_date
            while day < end_date:
                data_nans_removed.loc[day, station] = intercept + slope * data_nans_removed.loc[day, x_index]
                day = day + pd.Timedelta(1, 'd')
        return data_nans_removed


def find_best_correlations(df, md, lt, check_only_complete):
    to_fill = pd.DataFrame(columns=['x', 'usable'])

    for station, missing in md.items():
        if missing == {}:
            continue

        correlations = pd.DataFrame(columns=['correlation', 'p_value', 'usable'], index=[])

        for col in df.columns:
            if col == station or (check_only_complete and any((lt[station] & lt[col]))):
                continue
            data_for_regression = df.loc[df.index[~(lt[station] | lt[col])]][[station, col]]
            corr, p_value = scipy.stats.pearsonr(data_for_regression[col], data_for_regression[station])
            df2 = pd.DataFrame([[corr, p_value, data_for_regression.index]],
                               columns=['correlation', 'p_value', 'usable'], index=[col])
            correlations = pd.concat([correlations, df2])

        st_max_corr = correlations['correlation'].idxmax()

        if correlations['p_value'].loc[st_max_corr] >= 0.05:
            print("P-Value is too large")

        df3 = pd.DataFrame([[st_max_corr, correlations.loc[st_max_corr, 'usable']]], columns=['x', 'usable'],
                           index=[station])
        to_fill = pd.concat([to_fill, df3])
    return to_fill


def fill_data(df):
    missing_data = df.apply(consecutive_nans, 0)

    lookup_table = df.isna()

    to_fill = find_best_correlations(df, missing_data, lookup_table, False)

    df_no_nans = calculate_missing_values(df, to_fill, missing_data)

    print((df_no_nans.isna().sum() == 0).sum())

    missing_data = df_no_nans.apply(consecutive_nans, 0)
    lookup_table = df_no_nans.isna()

    to_fill = find_best_correlations(df_no_nans, missing_data, lookup_table, True)

    df_no_nans = calculate_missing_values(df_no_nans, to_fill, missing_data)
    return df_no_nans


data = read_in_csv()

data_no_nans = fill_data(data)

print((data_no_nans.isna().sum() == 0).sum())
