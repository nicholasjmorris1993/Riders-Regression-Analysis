import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, KBinsDiscretizer, PolynomialFeatures
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kstest
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


class Regression:
    def __init__(
        self, 
        name="Regression Analysis", 
        frac=1, 
        rename=True, 
        time=True, 
        binary=True, 
        imputation=True, 
        variance=True,
        scale=True,
        atwood=True,
        binning=True,
        reciprocal=True, 
        interaction=True, 
        selection=True,
    ):
        self.name = name  # name of the analysis
        self.frac= frac  # fraction of data to use for training the preprocessors
        self.rename = rename  # should features be renamed to remove whitespace?
        self.time = time  # should datetime features be computed?
        self.binary = binary  # should categorical features be converted to binary features?
        self.imputation = imputation  # should missing values be filled in?
        self.variance = variance  # should we remove constant features?
        self.scale = scale  # should we scale the features?
        self.atwood = atwood  # should we compute atwood numbers?
        self.binning = binning  # should we put continous features into bins?
        self.reciprocal = reciprocal  # should reciporcals be computed?
        self.interaction = interaction  # should interactions be computed?
        self.selection = selection  # should we perform feature selection?

        # create folders for output files
        self.folder(name)
        self.folder(f"{name}/dump")  # machine learning pipeline and data
        self.folder(f"{name}/plots")  # html figures

    def fit(self, X, y):
        # split up the data into training and testing
        trainX = X.head(int(0.8 * X.shape[0]))
        trainy = y.head(int(0.8 * y.shape[0]))
        testX = X.tail(int(0.2 * X.shape[0])).reset_index(drop=True)
        testy = y.tail(int(0.2 * y.shape[0])).reset_index(drop=True)
        
        # set aside data for preprocessing
        preprocessX = trainX.head(int(self.frac * trainX.shape[0]))
        preprocessy = trainy.head(int(self.frac * trainy.shape[0]))

        print("1/6) Model Training")
        start = time.time()

        # set up the machine learning pipeline
        self.names1 = FeatureNames(self.rename)
        self.datetime = TimeFeatures(self.time)
        self.categorical = CategoricalFeatures(self.binary)
        self.names2 = FeatureNames(self.rename)
        self.impute = ImputeFeatures(self.imputation)
        self.constant1 = ConstantFeatures(self.variance)
        self.scaler = ScaleFeatures(self.scale)
        self.selection1 = FeatureSelector(self.selection)
        self.numbers = AtwoodNumbers(self.atwood)
        self.bin = BinFeatures(self.binning)
        self.reciprocals = Reciprocals(self.reciprocal)
        self.interactions = Interactions(self.interaction)
        self.constant2 = ConstantFeatures(self.variance)
        self.selection2 = FeatureSelector(self.selection)
        self.tree = XGBRegressor(
            booster="gbtree",
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
        )
        
        # run preprocessing
        preprocessX = self.names1.fit_transform(preprocessX)
        preprocessX = self.datetime.fit_transform(preprocessX)
        preprocessX = self.categorical.fit_transform(preprocessX)
        preprocessX = self.names2.fit_transform(preprocessX)
        preprocessX = self.impute.fit_transform(preprocessX)
        preprocessX = self.constant1.fit_transform(preprocessX)
        preprocessX = self.scaler.fit_transform(preprocessX)
        preprocessX = self.selection1.fit_transform(preprocessX, preprocessy)
        numbers = self.numbers.fit_transform(preprocessX)
        preprocessX = self.bin.fit_transform(preprocessX)
        preprocessX = self.reciprocals.fit_transform(preprocessX)
        preprocessX = self.interactions.fit_transform(preprocessX)
        preprocessX = pd.concat([preprocessX, numbers], axis="columns")
        preprocessX = self.constant2.fit_transform(preprocessX)
        preprocessX = self.selection2.fit_transform(preprocessX, preprocessy)
        
        # run the pipeline on training data
        print("> Transforming The Training Data")
        trainX = self.names1.transform(trainX)
        trainX = self.datetime.transform(trainX)
        trainX = self.categorical.transform(trainX)
        trainX = self.names2.transform(trainX)
        trainX = self.impute.transform(trainX)
        trainX = self.constant1.transform(trainX)
        trainX = self.scaler.transform(trainX)
        trainX = self.selection1.transform(trainX)
        numbers = self.numbers.transform(trainX)
        trainX = self.bin.transform(trainX)
        trainX = self.reciprocals.transform(trainX)
        trainX = self.interactions.transform(trainX)
        trainX = pd.concat([trainX, numbers], axis="columns")
        trainX = self.constant2.transform(trainX)
        trainX = self.selection2.transform(trainX)
        print("> Training XGBoost")
        self.tree.fit(trainX, trainy)

        end = time.time()
        self.run_time(start, end)

        print("2/6) Model Performance")
        start = time.time()

        # transform the testing data and score the performance
        print("> Transforming The Testing Data")
        testX = self.names1.transform(testX)
        testX = self.datetime.transform(testX)
        testX = self.categorical.transform(testX)
        testX = self.names2.transform(testX)
        testX = self.impute.transform(testX)
        testX = self.constant1.transform(testX)
        testX = self.scaler.transform(testX)
        testX = self.selection1.transform(testX)
        numbers = self.numbers.transform(testX)
        testX = self.bin.transform(testX)
        testX = self.reciprocals.transform(testX)
        testX = self.interactions.transform(testX)
        testX = pd.concat([testX, numbers], axis="columns")
        testX = self.constant2.transform(testX)
        testX = self.selection2.transform(testX)
        self.performance(testX, testy)

        end = time.time()
        self.run_time(start, end)

        print("3/6) Model Deployment")
        start = time.time()

        # run the pipeline on all the data
        print("> Transforming All The Data")
        X = self.names1.transform(X)
        X = self.datetime.transform(X)
        X = self.categorical.transform(X)
        X = self.names2.transform(X)
        X = self.impute.transform(X)
        X = self.constant1.transform(X)
        X = self.scaler.transform(X)
        X = self.selection1.transform(X)
        numbers = self.numbers.transform(X)
        X = self.bin.transform(X)
        X = self.reciprocals.transform(X)
        X = self.interactions.transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.transform(X)
        X = self.selection2.transform(X)
        print("> Training XGBoost")
        self.tree.fit(X, y)

        end = time.time()
        self.run_time(start, end)

        print("4/6) Model Indicators")
        start = time.time()

        self.importance(X)

        end = time.time()
        self.run_time(start, end)

        # data we deployed on
        self.X = X
        self.y = y

    def predict(self, X):
        print("5/6) Model Prediction")
        start = time.time()

        # transform and predict new data
        print("> Transforming The New Data")
        X = self.names1.transform(X)
        X = self.datetime.transform(X)
        X = self.categorical.transform(X)
        X = self.names2.transform(X)
        X = self.impute.transform(X)
        X = self.constant1.transform(X)
        X = self.scaler.transform(X)
        X = self.selection1.transform(X)
        numbers = self.numbers.transform(X)
        X = self.bin.transform(X)
        X = self.reciprocals.transform(X)
        X = self.interactions.transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.transform(X)
        X = self.selection2.transform(X)
        y = self.tree.predict(X)

        end = time.time()
        self.run_time(start, end)

        print("6/6) Model Monitoring")
        start = time.time()

        self.monitor(X, y)

        end = time.time()
        self.run_time(start, end)

        return y
    
    def refit(self, X, y):
        print("Model Retraining:")
        start = time.time()

        # transform the new data
        print("> Transforming The New Data")
        X = self.names1.transform(X)
        X = self.datetime.transform(X)
        X = self.categorical.transform(X)
        X = self.names2.transform(X)
        X = self.impute.transform(X)
        X = self.constant1.transform(X)
        X = self.scaler.transform(X)
        X = self.selection1.transform(X)
        numbers = self.numbers.transform(X)
        X = self.bin.transform(X)
        X = self.reciprocals.transform(X)
        X = self.interactions.transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.transform(X)
        X = self.selection2.transform(X)
        
        # add the new data to the model data
        self.X = pd.concat([self.X, X], axis="index").reset_index(drop=True)
        self.y = pd.concat([self.y, y], axis="index").reset_index(drop=True)
        
        print("> Training XGBoost")
        self.tree.fit(self.X, self.y)
        
        end = time.time()
        self.run_time(start, end)

    def performance(self, X, y):
        # compute RMSE and R2
        predictions = self.tree.predict(X)
        y = y.iloc[:,0].to_numpy()
        self.bootstrap(y, predictions)
        df = pd.DataFrame({
            "RMSE": self.rmse,
            "R2": self.r2,
        })
        self.rmse = np.mean(self.rmse)
        self.r2 = np.mean(self.r2)

        # plot RMSE and R2
        self.histogram(
            df,
            x="RMSE",
            bins=20,
            title=f"{self.name}: Histogram For RMSE",
            font_size=16,
        )
        self.histogram(
            df,
            x="R2",
            bins=20,
            title=f"{self.name}: Histogram For R2",
            font_size=16,
        )

        # compute control limits for residuals
        error = y - predictions
        df = self.imr(error)

        # plot the control limits for residuals
        in_control = df.loc[(df["Individual"] >= df["Individual LCL"]) & (df["Individual"] <= df["Individual UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        self.histogram(
            df,
            x="Individual",
            vlines=[df["Individual LCL"][0], df["Individual UCL"][0]],
            bins=20,
            title=f"{self.name}: Histogram For Residuals, {in_control} In Control",
            font_size=16,
        )
        self.in_control = in_control

        # plot the control limits for the moving range of residuals
        in_control = df.loc[(df["Moving Range"] >= df["Moving Range LCL"]) & (df["Moving Range"] <= df["Moving Range UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        self.histogram(
            df,
            x="Moving Range",
            vlines=[df["Moving Range LCL"][0], df["Moving Range UCL"][0]],
            bins=20,
            title=f"{self.name}: Histogram For The Moving Range Of Residuals, {in_control} In Control",
            font_size=16,
        )

        # plot the predictions
        df = pd.DataFrame({
            "Prediction": predictions,
            "Actual": y,
        })
        self.parity(
            df,
            predict="Prediction",
            actual="Actual",
            title=f"{self.name}: Parity Plot",
            font_size=16,
        )

    def importance(self, X):
        # get the feature importance to determine indicators of the target
        importance = self.tree.feature_importances_
        indicators = pd.DataFrame({
            "Indicator": X.columns,
            "Importance": importance,
        })
        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]

        # plot the feature importance
        self.bar_plot(
            indicators,
            x="Indicator",
            y="Importance",
            title=f"{self.name}: XGBoost Feature Importance",
            font_size=16,
        )
        self.indicators = indicators

    def monitor(self, X, y):
        y_name = self.y.columns[0]
        X[y_name] = y  # new data
        df = pd.concat([self.X, self.y], axis="columns")  # data we trained on

        # see if the distribtuion of the new data is the same as the data we trained on
        pvalues = list()
        for column in df.columns:
            pvalues.append(kstest(
                df[column].tolist(),
                X[column].tolist(),
            ).pvalue)
        pvalues = pd.DataFrame({
            "Feature": df.columns,
            "pvalue": pvalues,
        })
        pvalues = pvalues.sort_values(
            by="pvalue", 
            ascending=False,
        ).reset_index(drop=True)

        # plot the pvalues
        self.bar_plot(
            pvalues,
            x="Feature",
            y="pvalue",
            title=f"{self.name}: Feature Drift, Drift Detected If pvalue < 0.05",
            font_size=16,
        )
        self.drift = pvalues

        # compute control limits for predictions
        df = self.imr(y)

        # plot the control limits for predictions
        in_control = df.loc[(df["Individual"] >= df["Individual LCL"]) & (df["Individual"] <= df["Individual UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        self.histogram(
            df,
            x="Individual",
            vlines=[df["Individual LCL"][0], df["Individual UCL"][0]],
            bins=20,
            title=f"{self.name}: Histogram For Predictions, {in_control} In Control",
            font_size=16,
        )

        # plot the control limits for the moving range of predictions
        in_control = df.loc[(df["Moving Range"] >= df["Moving Range LCL"]) & (df["Moving Range"] <= df["Moving Range UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        self.histogram(
            df,
            x="Moving Range",
            vlines=[df["Moving Range LCL"][0], df["Moving Range UCL"][0]],
            bins=20,
            title=f"{self.name}: Histogram For The Moving Range Of Predictions, {in_control} In Control",
            font_size=16,
        )

    def bootstrap(self, y_true, y_pred):
        df = pd.DataFrame({
            "Actual": y_true,
            "Predict": y_pred,
        })

        self.rmse = list()
        self.r2 = list()
        np.random.seed(0)
        seeds = np.random.random_integers(low=0, high=1e6, size=1000)

        # randomly sample RMSE and R2 scores
        for i in range(1000):
            sample = df.sample(frac=0.5, replace=True, random_state=seeds[i])
            self.rmse.append(mean_squared_error(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
                squared=False,
            ))
            self.r2.append(r2_score(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
            ))

    def imr(self, x: list):
        # control chart constants
        d2 = 1.128
        D4 = 3.267
        D3 = 0

        data = pd.DataFrame({
            "Observation": np.arange(len(x)) + 1,
            "Individual": x,
        })
        data["Moving Range"] = data["Individual"].diff().abs()
        data = data.dropna().reset_index(drop=True)

        # center lines
        Xbar = data["Individual"].mean()
        MRbar = data["Moving Range"].mean()

        # control limits
        I_UCL = Xbar + 3*MRbar / d2
        I_LCL = Xbar - 3*MRbar / d2
        I_CL = Xbar

        MR_UCL = MRbar*D4
        MR_LCL = MRbar*D3
        MR_CL = MRbar

        # results
        df = data[["Observation"]].copy()
        df["Individual"] = data["Individual"]
        df["Individual UCL"] = I_UCL
        df["Individual LCL"] = I_LCL
        df["Individual CL"] = I_CL
        df["Moving Range"] = data["Moving Range"]
        df["Moving Range UCL"] = MR_UCL
        df["Moving Range LCL"] = MR_LCL
        df["Moving Range CL"] = MR_CL

        return df

    def parity(self, df, predict, actual, color=None, title="Parity Plot", font_size=None):
        fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
        fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.name}/plots/{title}.html")
    
    def histogram(self, df, x, bins=20, vlines=None, title="Histogram", font_size=None):
        bin_size = (df[x].max() - df[x].min()) / bins
        fig = px.histogram(df, x=x, title=title)
        if vlines is not None:
            for line in vlines:
                fig.add_vline(x=line)
        fig.update_traces(xbins=dict( # bins used for histogram
                size=bin_size,
            ))
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.name}/plots/{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.name}/plots/{title}.html")

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)

    def folder(self, name):
        if not os.path.isdir(name):
            os.mkdir(name)

    def dump(self):
        # save the machine learning pipeline and data
        with open(f"{self.name}/dump/names1", "wb") as f:
            pickle.dump(self.names1, f)
        with open(f"{self.name}/dump/datetime", "wb") as f:
            pickle.dump(self.datetime, f)
        with open(f"{self.name}/dump/categorical", "wb") as f:
            pickle.dump(self.categorical, f)
        with open(f"{self.name}/dump/names2", "wb") as f:
            pickle.dump(self.names2, f)
        with open(f"{self.name}/dump/impute", "wb") as f:
            pickle.dump(self.impute, f)
        with open(f"{self.name}/dump/constant1", "wb") as f:
            pickle.dump(self.constant1, f)
        with open(f"{self.name}/dump/scaler", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{self.name}/dump/selection1", "wb") as f:
            pickle.dump(self.selection1, f)
        with open(f"{self.name}/dump/numbers", "wb") as f:
            pickle.dump(self.numbers, f)
        with open(f"{self.name}/dump/bin", "wb") as f:
            pickle.dump(self.bin, f)
        with open(f"{self.name}/dump/reciprocals", "wb") as f:
            pickle.dump(self.reciprocals, f)
        with open(f"{self.name}/dump/interactions", "wb") as f:
            pickle.dump(self.interactions, f)
        with open(f"{self.name}/dump/constant2", "wb") as f:
            pickle.dump(self.constant2, f)
        with open(f"{self.name}/dump/selection2", "wb") as f:
            pickle.dump(self.selection2, f)
        with open(f"{self.name}/dump/tree", "wb") as f:
            pickle.dump(self.tree, f)
        self.X.to_csv(f"{self.name}/dump/X.csv", index=False)
        self.y.to_csv(f"{self.name}/dump/y.csv", index=False)

    def load(self):
        # load the machine learning pipeline and data
        with open(f"{self.name}/dump/names1", "rb") as f:
            self.names1 = pickle.load(f)
        with open(f"{self.name}/dump/datetime", "rb") as f:
            self.datetime = pickle.load(f)
        with open(f"{self.name}/dump/categorical", "rb") as f:
            self.categorical = pickle.load(f)
        with open(f"{self.name}/dump/names2", "rb") as f:
            self.names2 = pickle.load(f)
        with open(f"{self.name}/dump/impute", "rb") as f:
            self.impute = pickle.load(f)
        with open(f"{self.name}/dump/constant1", "rb") as f:
            self.constant1 = pickle.load(f)
        with open(f"{self.name}/dump/scaler", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{self.name}/dump/selection1", "rb") as f:
            self.selection1 = pickle.load(f)
        with open(f"{self.name}/dump/numbers", "rb") as f:
            self.numbers = pickle.load(f)
        with open(f"{self.name}/dump/bin", "rb") as f:
            self.bin = pickle.load(f)
        with open(f"{self.name}/dump/reciprocals", "rb") as f:
            self.reciprocals = pickle.load(f)
        with open(f"{self.name}/dump/interactions", "rb") as f:
            self.interactions = pickle.load(f)
        with open(f"{self.name}/dump/constant2", "rb") as f:
            self.constant2 = pickle.load(f)
        with open(f"{self.name}/dump/selection2", "rb") as f:
            self.selection2 = pickle.load(f)
        with open(f"{self.name}/dump/tree", "rb") as f:
            self.tree = pickle.load(f)
        self.X = pd.read_csv(f"{self.name}/dump/X.csv")
        self.y = pd.read_csv(f"{self.name}/dump/y.csv")


class FeatureNames:
    def __init__(self, rename=True):
        self.rename = rename

    def fit(self, X, y=None):
        if not self.rename:
            return self
        print("> Renaming Features")

        self.columns = [re.sub(" ", "_", col) for col in X.columns]
        return self

    def transform(self, X, y=None):
        if not self.rename:
            return X

        X.columns = self.columns
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

        
class TimeFeatures:
    def __init__(self, time=True):
        self.time = time

    def fit(self, X, y=None):
        if not self.time:
            return self
        print("> Extracting Time Features")

        # check if any columns are timestamps
        self.features = [col for col in X.columns if is_datetime(X[col])]
        return self

    def transform(self, X, y=None):
        if not self.time:
            return X

        if len(self.features) == 0:
            return X
        else:
            # convert any timestamp columns to datetime data type
            df = X.copy().apply(
                lambda col: pd.to_datetime(col, errors="ignore")
                if col.dtypes == object 
                else col, 
                axis=0,
            )

            # extract timestamp features
            dt = pd.DataFrame()
            for col in self.features:
                dt[f"{col}_year"] = df[col].dt.isocalendar().year
                dt[f"{col}_month_of_year"] = df[col].dt.month
                dt[f"{col}_week_of_year"] = df[col].dt.isocalendar().week
                dt[f"{col}_day_of_month"] = df[col].dt.day
                dt[f"{col}_day_of_week"] = df[col].dt.dayofweek
                dt[f"{col}_hour_of_day"] = df[col].dt.hour
                dt[f"{col}_minute_of_hour"] = df[col].dt.minute
            dt = pd.concat([df.drop(columns=self.features), dt], axis="columns")
            return dt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class CategoricalFeatures:
    def __init__(self, binary=True):
        self.binary = binary

    def fit(self, X, y=None):
        if not self.binary:
            return self
        print("> Transforming Categorical Features")

        strings = X.select_dtypes(include="object").columns.tolist()
        df = X.copy().drop(columns=strings)
        numbers = [col for col in df.columns if len(df[col].unique()) <= 53]
        self.categorical = strings + numbers
        if len(self.categorical) == 0:
            return self
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return self.encoder.fit(X[self.categorical].astype(str))

    def transform(self, X, y=None):
        if not self.binary:
            return X

        if len(self.categorical) == 0:
            return X
        continuous = X.copy().drop(columns=self.categorical)
        binary = self.encoder.transform(X[self.categorical].astype(str))
        binary = pd.DataFrame(binary, columns=self.encoder.get_feature_names_out())
        df = pd.concat([continuous, binary], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ImputeFeatures:
    def __init__(self, imputation=True):
        self.imputation = imputation

    def fit(self, X, y=None):
        if not self.imputation:
            return self
        print("> Filling In Missing Values")

        self.columns = X.columns
        self.imputer = KNNImputer()
        return self.imputer.fit(X)

    def transform(self, X, y=None):
        if not self.imputation:
            return X

        df = self.imputer.transform(X)
        df = pd.DataFrame(df, columns=self.columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ConstantFeatures:
    def __init__(self, variance=True):
        self.variance = variance

    def fit(self, X, y=None):
        if not self.variance:
            return self
        print("> Removing Constant Features")

        self.selector = VarianceThreshold()
        return self.selector.fit(X)

    def transform(self, X, y=None):
        if not self.variance:
            return X

        df = self.selector.transform(X)
        df = pd.DataFrame(df, columns=self.selector.get_feature_names_out())
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ScaleFeatures:
    def __init__(self, scale=True):
        self.scale = scale

    def fit(self, X, y=None):
        if not self.scale:
            return self
        print("> Scaling Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.scaler = MinMaxScaler(feature_range=(0.2, 0.8))
        return self.scaler.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.scale:
            return X

        if len(self.columns) == 0:
            return X
        df = self.scaler.transform(X[self.columns])
        df = pd.DataFrame(df, columns=self.columns)
        df = pd.concat([X.drop(columns=self.columns), df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    
class AtwoodNumbers:
    def __init__(self, atwood=True):
        self.atwood = atwood

    def fit(self, X, y=None):
        if not self.atwood:
            return self
        print("> Computing Atwoood Numbers")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.atwood:
            return pd.DataFrame()

        if len(self.columns) < 2:
            return pd.DataFrame()
        numbers = list()
        pairs = list(combinations(self.columns, 2))
        for pair in pairs:
            numbers.append(pd.DataFrame({
                f"({pair[0]}-{pair[1]})/({pair[0]}+{pair[1]})": (X[pair[0]] - X[pair[1]]) / (X[pair[0]] + X[pair[1]]),
            }))
        df = pd.concat(numbers, axis="columns")
        df = df.fillna(0)
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    

class BinFeatures:
    def __init__(self, binning=True):
        self.binning = binning

    def fit(self, X, y=None):
        if not self.binning:
            return self
        print("> Binning Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.binner = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform", subsample=None)
        return self.binner.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.binning:
            return X

        if len(self.columns) == 0:
            return X
        df = self.binner.transform(X[self.columns]).toarray()
        edges = self.binner.bin_edges_
        columns = list()
        for i, feature in enumerate(self.columns):
            bins = np.around(edges[i], 6)
            columns.append(f"{feature}({bins[0]}-{bins[1]})")
            columns.append(f"{feature}({bins[1]}-{bins[2]})")
            columns.append(f"{feature}({bins[2]}-{bins[3]})")
        df = pd.DataFrame(df, columns=columns)
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    
class Reciprocals:
    def __init__(self, reciprocal=True):
        self.reciprocal = reciprocal

    def fit(self, X, y=None):
        if not self.reciprocal:
            return self
        print("> Computing Reciprocals")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.reciprocal:
            return X

        if len(self.columns) == 0:
            return X
        df = 1 / X.copy()[self.columns]
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        df.columns = [f"1/{col}" for col in df.columns]
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class Interactions:
    def __init__(self, interaction=True):
        self.interaction = interaction

    def fit(self, X, y=None):
        if not self.interaction:
            return self
        print("> Computing Interactions")

        self.interactions = PolynomialFeatures(
            degree=2, 
            interaction_only=True, 
            include_bias=False,
        )
        return self.interactions.fit(X)

    def transform(self, X, y=None):
        if not self.interaction:
            return X

        df = self.interactions.transform(X)
        columns = self.interactions.get_feature_names_out()
        columns = [re.sub(" ", "*", col) for col in columns]
        df = pd.DataFrame(df, columns=columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class FeatureSelector:
    def __init__(self, selection=True):
        self.selection = selection

    def fit(self, X, y=None):
        if not self.selection:
            return self
        print("> Selecting Features")

        tree = XGBRegressor(
            booster="gbtree",
            n_estimators=25, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=0,
            n_jobs=-1,
        )
        tree.fit(X, y)

        # get the feature importance to determine indicators of the target
        importance = tree.feature_importances_
        indicators = pd.DataFrame({
            "Indicator": X.columns,
            "Importance": importance,
        })
        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]
        self.columns = indicators["Indicator"].tolist()

        return self

    def transform(self, X, y=None):
        if not self.selection:
            return X

        return X[self.columns]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
