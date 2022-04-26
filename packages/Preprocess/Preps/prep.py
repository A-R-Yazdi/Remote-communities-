import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from math import sqrt
import scipy.stats as scs
import numpy as np
import seaborn as sns
import statsmodels.api as sm


class Preprocessing:
    def __init__(self):
        pass

    def create_key(self, keyString="df", keyRange=9):
        """create a list of dataframe names, key=['df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'df9']
        keyString is the invariable in the key, e.g. 'df' in 'df1'
        """
        key = []
        for i in range(9):
            key.append(keyString + str(i + 1))
        return key

    dfc = None

    def create_valuelist(self, DF=dfc, col="lat"):
        list_of_df = []
        col_val_list = list(DF[col].value_counts().keys())
        # print(col_val_list)
        for col_val in col_val_list:
            list_of_df.append(DF[DF[col] == col_val].reset_index(drop=True))
        return list_of_df

    def load_n_sort(self, df="df2", data_root=None):
        df = pd.read_csv(data_root + df + ".csv", index_col=0)
        df = df.sort_values(by="datetime", ascending=True)
        return df

    def load_dframes(
        self, df_name="df", prop="", df_enum_range=range(1, 10), data_root=None
    ):
        # print('preloading')
        if isinstance(df_enum_range, range) and isinstance(df_name, str):
            # print("loading....")
            list_of_df = []
            for i in df_enum_range:
                n = df_name + str(i) + prop
                # use globals() to convert a string into var name so that we can use later
                vars()[n] = self.load_n_sort(n, data_root=data_root)
                list_of_df.append(vars()[n])

        return list_of_df

    def show_missing_datetime(
        self,
        diff_from: pd.DataFrame,
        start_d="2013-01-01",
        end_d="2016-01-01",
        freq="H",
    ):
        """input var diff_from has to be a DataFrame with no duplications"""
        if (
            isinstance(diff_from, pd.DataFrame)
            and len(diff_from[diff_from.duplicated()]) == 0
        ):
            diff_from["datetime"] = pd.to_datetime(diff_from["datetime"])
            diff = pd.date_range(start=start_d, end=end_d, freq=freq)[:-1].difference(
                diff_from["datetime"]
            )
            return diff

    def fill_missing_rows(self, df=None):
        """input must be a dataframe"""
        if isinstance(df, pd.DataFrame):
            df["datetime"] = pd.to_datetime(df["datetime"])
            additional_df = pd.DataFrame(
                index=self.show_missing_datetime(diff_from=df), columns=df.columns
            )
            df = (
                df.set_index("datetime")
                .append(additional_df)
                .sort_index(ascending=True)
                .ffill()
                .reset_index()
            )
            df["datetime"] = df["index"]

        return df.drop(["index"], axis=1)

    def show_shape(self, dfs: List[pd.DataFrame]):
        li = list(dfs)
        shape = []
        for i in range(len(li)):
            if not isinstance(li[i], pd.DataFrame):
                print("input needs to be a list of dataframes")
            shape.append(li[i].shape)

        df = pd.DataFrame(
            shape,
            columns=["num_of_rows", "num_of_cols"],
            index=["dataframe " + str(i) for i in range(len(li))],
        )
        return df

    def show_outliers(
        self, df=None, col_name="kw_cap", num_of_std=3, anomaly_window=24 * 365 * 3
    ):
        """
        df needs to be a DataFrame, col_name needs to be a string, num_of_std is an integer
        num_of_std decides the window of NORMAL values
        anomaly_window is the window size from which we decide the mean, std
        """
        # if isinstance(df, pd.DataFrame) & isinstance(col_name, str) & isinstance(num_of_std, int):
        #   pass
        # else:
        #   print("params need to be the correct type")

        d_outliers = pd.DataFrame(columns=df.columns)

        # if the row number is less than anomaly_window, take the first anomaly_window number of rows and calculate the mean, std
        dd = df[:anomaly_window]
        d_mean = dd[col_name].mean()
        d_std = dd[col_name].std()
        low, high = d_mean - (num_of_std * d_std), d_mean + (num_of_std * d_std)
        d_outliers = pd.concat(
            [d_outliers, dd[(dd[col_name] < low) | (dd[col_name] > high)]]
        )

        # if anomaly_window < len(df), this means d_mean, d_std changes row by row after we go beyond row number anomaly_window
        dd = df[anomaly_window:]
        for row in range(anomaly_window, len(df)):
            d_mean = df[col_name][row - anomaly_window : row].mean()
            d_std = df[col_name][row - anomaly_window : row].std()
            low, high = d_mean - (num_of_std * d_std), d_mean + (num_of_std * d_std)
            if (dd[col_name][row] < low) | (dd[col_name][row] > high):
                d_outliers = pd.concat([d_outliers, df[row : row + 1]])

        return d_outliers

    def replace_outliers(
        self,
        df=None,
        col_name: str = "kw_cap",
        num_of_std: int = 3,
        anomaly_window: int = 24 * 365 * 3,
    ):
        """
        df needs to be a DataFrame, col_name needs to be a string, num_of_std is an integer
        num_of_std decides the window of NORMAL values
        anomaly_window is the window size from which we decide the mean, std
        """
        # index is the outliers row number
        index = self.show_outliers(
            df=df,
            col_name=col_name,
            anomaly_window=anomaly_window,
            num_of_std=num_of_std,
        ).index

        for i in index:

            if i < anomaly_window:
                # if the row number is less than anomaly_window, take the first anomaly_window number of rows and calculate the mean, std
                dd = df[:anomaly_window]
                d_mean = dd[col_name].mean()
                d_std = dd[col_name].std()
                low, high = d_mean - (num_of_std * d_std), d_mean + (num_of_std * d_std)
                df[col_name][i] = d_mean

            else:
                dd = df[anomaly_window:]
                for row in range(anomaly_window, len(df)):
                    d_mean = df[col_name][row - anomaly_window : row].mean()
                    d_std = df[col_name][row - anomaly_window : row].std()
                    low, high = (
                        d_mean - (num_of_std * d_std),
                        d_mean + (num_of_std * d_std),
                    )
                    df[col_name][i] = d_mean

        return df

    def plot_dfs(self, dfs: List[pd.DataFrame]):
        li = list(dfs)

        for i in range(len(li)):
            data = li[i][["datetime", "kw_cap"]].set_index("datetime")
            fig = data.plot(figsize=(20, 4), title="df" + str(i + 1) + " kw_cap")
            # fig.savefig(fig_root+'df'+str(i)+'.png')
            # print(fig)

            temp_data = li[i][["datetime", "Temp (°C)"]].set_index("datetime")
            fig = temp_data.plot(
                figsize=(20, 4), title="df" + str(i + 1) + " temperature"
            )
            # use a print() statement will print: 'AxesSubplot' object ......(something something)
            # print(fig)
            plt.show()

    def scale_series(
        self,
        dfs: List[pd.DataFrame],
        cols=[str],
        cols_scaler_range=[tuple],
        DEBUG_MODE=False,
    ):
        """
        cols are a list strings, they are the dataframe column names; 
        cols_scaler_range are a list of tuples, they are the range we want to scale the cols to.  
        cols_scaler_range must have the same length as cols.
        """
        assert all(isinstance(s, str) for s in cols)
        assert all(isinstance(s, tuple) for s in cols_scaler_range)
        li = list(dfs)
        scaled_df = []
        for i in range(len(li)):
            dic = {}
            for j in range(len(cols)):
                data = li[i][cols[j]].values
                # reshape to (n_sample, n_features)
                data = data.reshape(len(data), 1)
                data_scaler = MinMaxScaler(feature_range=cols_scaler_range[j])
                data_scaler = data_scaler.fit(data)
                data_normalized = pd.Series(
                    data_scaler.transform(data).reshape(len(data))
                )
                dic[cols[j]] = data_normalized

                if DEBUG_MODE:
                    print(
                        cols[j]
                        + " Min: %f, Max: %f"
                        % (data_scaler.data_min_, data_scaler.data_max_)
                    )
                    print()

            # Concatenating the series side by side as depicted by axis=1
            scaled_df.append(pd.concat(dic, axis=1))

        return scaled_df

    def standardize_series(
        self, dfs: List[pd.DataFrame], cols: [str], DEBUG_MODE=False
    ):
        """
        cols are a list strings, they are the dataframe column names; 
        cols_scaler_range are a list of tuples, they are the range we want to scale the cols to.  
        cols_scaler_range must have the same length as cols.
        """
        assert all(isinstance(s, str) for s in cols)
        li = list(dfs)
        stdard_df = []
        for i in range(len(li)):
            dic = {}
            for j in range(len(cols)):
                data = li[i][cols[j]].values
                data = data.reshape(len(data), 1)

                scaler = StandardScaler()
                scaler = scaler.fit(data)

                if DEBUG_MODE:
                    print(
                        "Mean: %f, StandardDeviation: %f"
                        % (scaler.mean_, sqrt(scaler.var_))
                    )

                normalized = scaler.transform(data)
                data_normalized = pd.Series(scaler.transform(data).reshape(len(data)))

                dic[cols[j]] = data_normalized
            # Concatenating the series side by side as depicted by axis=1
            stdard_df.append(pd.concat(dic, axis=1))

            print()

        return stdard_df

    def plot_QQ(self, df: pd.DataFrame, col: str):
        r_range = np.linspace(min(df[col].dropna()), max(df[col].dropna()), num=1000)
        mu = df[col].dropna().mean()
        sigma = df[col].dropna().std()
        norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

        # Plot the histogram and the Q-Q plot
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

        # distplot() deprecated, use displot() or histplot()
        # sns.distplot(df[col].dropna(), kde=False, norm_hist=True, ax=ax[0])
        # sns.histplot(df[col].dropna(), kde=False, ax=ax[0])
        sns.displot(df[col].dropna())
        ax[0].set_title("Distribution of kw_cap", fontsize=16)
        ax[0].plot(r_range, norm_pdf, "g", lw=2, label=f"N({mu:.2f}, {sigma**2:.4f})")
        ax[0].legend(loc="upper left")

        # Q-Q plot
        qq = sm.qqplot(df[col].dropna().values, line="s", ax=ax[1])
        ax[1].set_title("Q-Q plot", fontsize=16)
        plt.show()
