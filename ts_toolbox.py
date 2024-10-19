# %%
# all imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# %%
# Global Variables
J = 9
K = 9
LAGS = 20  # make sure that # lags is > (J + K)

np.random.seed(6313)
DEC = 3
def rnd(x): return np.around(x, decimals=DEC)


WN_Mean = 0
WN_Var = 1
NUM_OBS = 1000


def standardize(x): return (x - x.mean()) / x.std()
def roots(x): return np.roots(np.r_[1, x])


def deSeas(data, s):
    diff = []
    for i in range(s, len(data)):
        value = data[i] - data[i - s]
        diff.append(value)
    return np.array(diff)

# def unDeSeas(data, s):
#     diff = []
#     for i in range(s, len(data)):
#         value = data[i] + data[i - s]
#         diff.append(value)
#     return pd.DataFrame(diff, index=data.index)

# %%
# Correlation Coefficient


def calc_correlation_coefficent(x, y):
    ds_x = np.array(x)
    ds_y = np.array(y)

    xbar = ds_x.mean()
    ybar = ds_y.mean()
    nom = np.sum((ds_x - xbar) * (ds_y - ybar))
    if nom == 0:
        return 0
    den = np.sqrt(np.sum(np.square(ds_x - xbar))
                  * np.sum(np.square(ds_y - ybar)))
    return nom / den

# %%
# ACF


def mkDblSd(dl, lags):
    return range(-lags, lags+1), (dl[::-1] + dl[1:])


def acf(df, lags):
    ybar = df.mean()
    den = np.square(df - ybar).sum()
    T = len(df)

    res = []
    for r in range(lags + 1):
        nom = (df[r:] - ybar) * (df[:T - r] - ybar)
        res.append(nom.sum() / den)

    return res


def plotAcf(data, num_lags, title):
    # Plot ACF
    sig = 1.96 / np.sqrt(len(data))

    flattened_data = np.array(data).flatten()
    acf_val = acf(flattened_data, num_lags)
    x_a, y_a = mkDblSd(acf_val, num_lags)
    (m, s, b) = plt.stem(x_a, y_a)
    plt.setp(m, color='red')
    plt.axhspan(-sig, sig, alpha=0.3, color='blue')
    plt.title(f'ACF of {title}, #lags={num_lags}')
    plt.xlabel('#Lags')
    plt.ylabel('Magnitude')

    plt.show()
    return acf_val


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

# %%
# Forecast accuracy measures


def SSE(e):
    return e.T @ e


def MSE(e):
    return np.mean(np.square(e))


def RMSE(e):
    return np.sqrt(MSE(e))


def MAE(e):
    return np.mean(np.abs(e))


def forecastAccuracy(test, forecast, method):
    e = test.flatten() - forecast.flatten()
    # print(test.values.flatten().shape, holtf.values.flatten().shape)
    # print(e)
    print(f'{method} method, MSE: {rnd(MSE(e))} \n'
          f'RMSE: {rnd(RMSE(e))} \n'
          f'SSE: {rnd(SSE(e))} \n'
          f'MAE: {rnd(MAE(e))}')


ERR_HEAD = ['Forecast Method', 'MSE', 'RMSE', 'SSE', 'MAE']


def getAccuracy(test, forecast, method):
    e = test.flatten() - forecast.flatten()
    record = []
    record.append(method)
    record.append(f'{rnd(MSE(e))}')
    record.append(f'{rnd(RMSE(e))}')
    record.append(f'{rnd(SSE(e))}')
    record.append(f'{rnd(MAE(e))}')
    return record

# %%
# Drift Fill Data
#


def driftFill(data, old_index_column, new_index, columns):
    new_index = list(new_index)
    result = {}
    result[old_index_column] = new_index
    old_index = {}

    for i, v in enumerate(data[old_index_column]):
        old_index[v] = i

    for c in columns:
        initial = data[c][0]
        new_data = []
        for i, j in enumerate(new_index):
            if j in old_index:
                x = old_index[j]
                new_data.append(data[c][x])
            else:
                if i > 1:
                    new_item = new_data[i - 1] + \
                        (new_data[i - 1] - new_data[0]) / (i - 1)
                    new_data.append(new_item)
                else:
                    new_data.append(initial)

        result[c] = new_data
    return pd.DataFrame(result)

# %%
# Data Cleaning


def downSample(step, data, index, columns):
    ds_data = pd.DataFrame(columns=columns)
    total = len(data)
    dates = []
    for i in range(step, total, step):
        ds_data.loc[len(ds_data.index)] = data[columns][i-step:i].mean()
        dates.append(data[index][i])
    ds_data[index] = dates
    return ds_data

# %%
# Plot helpers

# Plot X vs Y with title and labels


def plotXY(X, Y, title, xlabel, ylabel, show=True, fs=(16, 8)):
    if fs != None:
        plt.figure(figsize=fs)
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if show:
        plt.show()

# Plot Train data and Test vs Prediction, with title, labels
# def plotTTP(train, test, prediction, title='Train w/ Test VS Prediction', xlabel='t', ylabel='y(t)'):


def plotForecast(train, test, forecast, method='', show=True, xlabel='t', ylabel='y(t)'):
    train_size = len(train)
    test_size = len(test)
    total = train_size + test_size

    plt.figure(figsize=(12, 4))
    plt.plot(train.index, train, label='Training Set')
    plt.plot(test.index, test, label='Test Set')
    plt.plot(test.index, forecast, label='Forecast')

    plt.title(f'{method} Forecast')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if show:
        plt.show()
    else:
        plt.tight_layout()

# Plot Table


def plotTable(cells, columns, title):
    plt.figure(linewidth=2, tight_layout={'pad': 1}, figsize=(8, 6))
    ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))

    the_table = plt.table(cellText=cells,
                          colColours=ccolors,
                          colLabels=columns, loc='center')
    the_table.scale(1, 1)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.box(on=None)
    plt.suptitle(title)
    plt.show()

# %%
# Stationarity Tests


def calculate_rolling_mean_var(data):
    rolling_means = []
    rolling_variances = []
    for i in range(len(data)):
        roll = data.head(i+1)
        rolling_means.append(roll.mean())
        rolling_variances.append(roll.var(ddof=0))
    return rolling_means, rolling_variances


def plot_rolling_mean_var(data, xlabel, ylabel):
    rolling_means = []
    rolling_variances = []
    for i in range(len(data)):
        rolling_means.append(np.mean(data[:i]))
        rolling_variances.append(np.var(data[:i]))
    # plot
    # x = range(len(data))
    x = data.index
    plt.subplot(2, 1, 1)
    plotXY(x, rolling_means,
           xlabel=xlabel, ylabel=ylabel, title='Rolling Mean',
           show=False, fs=None)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plotXY(x, rolling_variances,
           title='Rolling Variance', xlabel=xlabel, ylabel=ylabel,
           show=False, fs=None)
    plt.tight_layout()
    plt.show()


def printAdf(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def printKpss(x):
    print('Results of KPSS Test:')
    kpsstest = kpss(x, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=[
                            'Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)

# %%
# Residuals analysis (Chi-Square test)


def residualAnalysis(data, error, lags=LAGS):
    re = plotAcf(error.flatten(), LAGS, 'ACF of residuals')
    Q = len(data)*np.sum(np.square(re[LAGS:]))
    # Q = sm.stats.acorr_ljungbox(e.flatten(), lags=LAGS)
    DOF = LAGS - 6
    alfa = 0.01
    from scipy.stats import chi2
    chi_critical = chi2.ppf(1-alfa, DOF)

    if Q < chi_critical:
        print(f"The residual is white, Q: {Q}, chi_critical: {chi_critical}")
    else:
        print(
            f"The residual is NOT white, Q: {Q}, chi_critical: {chi_critical}")

# %%
# STL Decomposition


def STLDecompose(data, feature, dates, s=12, name='STL Decompse'):
    df_series = pd.Series(data=np.array(data[feature]),
                          index=pd.date_range(start=dates[0],
                                              freq='d',
                                              periods=len(data)),
                          name=name)

    stl_dcomp = STL(df_series, seasonal=s)
    result = stl_dcomp.fit()

    plt.figure(figsize=(8, 8))
    fig = result.plot()
    plt.xlabel('Timeline')
    plt.tight_layout()
    plt.show()

    T = result.trend
    S = result.seasonal
    R = result.resid

    seas_adj = data[feature] - np.array(S)
    detrended = data[feature] - np.array(T)

    plotXY(dates, detrended, xlabel='Date',
           ylabel=f'{feature}', title='Detrended Data', fs=(12, 4))
    plotXY(dates, seas_adj, xlabel='Date',
           ylabel=f'{feature}', title='Seasonally Adjusted Data', fs=(12, 4))

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
    print(f'The strength of trend for this data set is {100*F:.3f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is {100*FS:.3f}%')

# %%
# Basic Prediction Models


def getAvgModel(trn, tst):
    mean = trn.mean()

    hSteps = [trn[0]]
    for i in range(1, len(trn)):
        hSteps.append(trn[:i].mean())

    return np.full(shape=(len(tst)), fill_value=mean), np.array(hSteps)


def getNaiveModel(trn, tst):
    hSteps = [trn[0]]
    for i in range(1, len(trn)):
        hSteps.append(trn[i-1])

    return np.full(shape=(len(tst)), fill_value=trn[-1]), np.array(hSteps)


def getDriftModel(trn, tst):
    hSteps = [trn[0], trn[1]]
    for i in range(2, len(trn)):
        step = trn[i - 1] + (trn[i - 1] - trn[0]) / (i - 1)
        hSteps.append(step)

    forecasts = []
    increment = (trn[-1] - trn[0]) / (len(trn) - 1)
    for i in range(len(tst)):
        forecasts.append(trn[-1] + (i+1) * increment)

    return np.array(forecasts), np.array(hSteps)


def getSESModel(trn, tst, alpha=0.5):
    tSteps = [trn[0]]
    for i in range(1, len(trn)):
        step = alpha * trn[i - 1] + (1 - alpha) * tSteps[i-1]
        tSteps.append(step)

    frcst = alpha * trn[-1] + (1 - alpha) * tSteps[-1]

    return np.full(shape=(len(tst)), fill_value=frcst), np.array(tSteps)

# %%
# Holt Winters Model For Prediction


def predictHoltWinters(train, test, trend=None, damped=False, seasonal=None, seasonal_periods=None):
    holtt = ets.ExponentialSmoothing(
        train, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    return holtt.forecast(steps=len(test))

# plot holt winters.
# trend = 'mul' | 'add' | None,
# damped = True | False, only provide if trend is not None
# seasonal = 'mul' | 'add' | None,
# seasonal_periods is a number


def plotHoltWinters(train, test, trend=None, damped=False, seasonal=None, seasonal_periods=None, title='', xlabel='', ylabel=''):
    holtf = predictHoltWinters(
        train, test, trend, damped, seasonal, seasonal_periods)
    holtf = pd.DataFrame(holtf).set_index(test.index)

    # forecastAccuracy(test.values, holtf.values, 'Holt-Winters')
    plotForecast(train, test, holtf, title, xlabel=xlabel, ylabel=ylabel)
    return holtf

# %%
# SVD Decomposition:


def svdDecompose(data, ind_var):
    U, s, V = np.linalg.svd(data[ind_var])
    print("Singular Values:")
    print(s)

    # Condition Number:
    print(f'Condition Number: {np.linalg.cond(data[ind_var])}')

# %%
# Regression Models - 1. Backward Step Regression


def backStepReg(train, depVar, feats, use_adjr2=True):
    Y = np.array(train[depVar])
    X = np.array(train[feats])

    model = sm.OLS(Y, sm.add_constant(X)).fit()
    prev = model.rsquared_adj
    removed = []

    pval_df = pd.DataFrame()
    pval_df['pvals'] = model.pvalues[1:]
    pval_df['features'] = feats

    # print('Features:')
    # print(np.array(feats))
    # print(
    #     f'\nAIC: {model.aic:.2f}, BIC: {model.bic:.2f}, Adjusted_R2: {model.rsquared_adj:.4f} before removing')

    sorted_decreasing = pval_df.sort_values('pvals', ascending=False)

    for i, col in enumerate(sorted_decreasing['features']):
        cols = [x for x in sorted_decreasing['features']
                if x not in (removed + [col])]
        pval = sorted_decreasing['pvals'][i]
        model = sm.OLS(Y, sm.add_constant(train[cols].to_numpy())).fit()

        if use_adjr2 and model.rsquared_adj >= prev:
            removed.append(col)
        elif not (use_adjr2) and pval > 0.05:
            removed.append(col)
        if use_adjr2:
            print(f'Adjusted_R2: {model.rsquared_adj:.4f} - {col}')
        else:
            print(f'P-Value: {pval:.4f} - {col}')

    return removed


# %%
# Regression Models - 2. VIF Analysis
def vifMethod(train, depVar, feats, threshold=10):
    X = sm.add_constant(np.array(train[feats]))
    Y = np.array(train[depVar])

    model = sm.OLS(Y, X).fit()
    removed = []

    vif = pd.DataFrame()
    vif['columns'] = ['ones']+feats
    vif['vif'] = [variance_inflation_factor(
        X, i) for i in range(len(vif['columns']))]
    sorted_decreasing = vif.sort_values('vif', ascending=False)
    recommend = []

    # print('Question 8 Solution:')
    # print(f'{sorted_decreasing}\n')

    # print(f'\nAIC: {model.aic:.2f}, BIC: {model.bic:.2f}, Adjusted_R2: {model.rsquared_adj:.4f} before removing')

    for i, col in enumerate(sorted_decreasing['columns']):
        cols = [x for x in feats if x not in (removed + [col])]
        vif_val = sorted_decreasing['vif'][i]
        model = sm.OLS(Y, sm.add_constant(np.array(train[cols]))).fit()

        if vif_val > threshold and col != 'ones':
            recommend.append(col)

        removed.append(col)
        print(f'VIF: {vif_val:.4f} - {col}')

    return recommend

# %%
# Parameter prediction - 1. ARIMA
# trend = 'n' | 'c' (means constant) | 't' (means linear) | 'ct' (means constant and linear)
# arma = (na, nb)
# sarima = (na, d, nb, s)


def predictARIMA(y, na, nb, trend='n', seasonal=(0, 0, 0, 0)):
    # STEP 9
    model = sm.tsa.arima.ARIMA(
        y, order=(na, 0, nb), trend=trend, seasonal_order=seasonal).fit()
    print(f'AR Params: {rnd(model.arparams)}')
    print(f'MA Params: {rnd(model.maparams)}')

    # step 10 - highlight coefficients and confience interval in report
    print(model.summary())
    return model

# %%


def linRegPredict(train, test, dep, feats):
    model = sm.OLS(train[dep].to_numpy(),
                   sm.add_constant(
        train[feats].to_numpy())).fit()
    return model, model.predict(sm.add_constant(
        test[feats].to_numpy()))


def plotTestVPrediction(train, test, dep, feats, title, xlabel='t', ylabel='y(t)'):
    model, prediction = linRegPredict(train, test, dep, feats)
    prediction = pd.DataFrame(prediction).set_index(test.index)

    plt.figure(figsize=(12, 4))
    plt.plot(test[dep], label='Test Data')
    plt.plot(prediction, label='Prediction')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
    return model, prediction


def plotRegressPrediction(train, test, dep, feats, title, xlabel='t', ylabel='y(t)'):
    _, prediction = linRegPredict(train, test, dep, feats)
    prediction = pd.DataFrame(prediction).set_index(test.index)

    plt.figure(figsize=(12, 4))
    plotForecast(train[dep], test[dep], prediction,
                 title, xlabel=xlabel, ylabel=ylabel)
    return prediction


def showRegressionMetrics(model, error, m='Model Metrics'):
    # print(model.summary())
    print(f'\n{m}')
    # mse
    print(f'MSE: {MSE(error):.3f}')
    # rmse
    print(f'RMSE: {RMSE(error):.3f}')
    # AIC, BIC
    print(f'AIC: {model.aic:.3f}, BIC: {model.bic:.3f}')
    # R sq + Adj R sq
    print(f'R Sq: {model.rsquared:.3f}, Adjusted R Sq: {model.rsquared_adj:.3f}')


def f_t_test(model, m='F-test & T-test'):
    print(f'\n{m}')
    # f-test and t-test analysis
    print(f'final model t-test results: {np.around(model.pvalues, 3)}')
    print(f'final model f-test results: {np.around(model.f_pvalue, 3)}')

def m_v(data, m='Mean and Variance'):
    print(f'\n{m}')
    print(f'Mean: {np.mean(data):.3f}')
    print(f'Variance: {np.var(data):.3f}')

def residual_analysis(acf, lags, T, label='!!!'):
    Q = T*np.sum(np.square(acf[lags:]))
    # Q = sm.stats.acorr_ljungbox(e.flatten(), lags=LAGS)
    DOF = lags - 6
    alfa = 0.01
    from scipy.stats import chi2
    chi_critical = chi2.ppf(1-alfa, DOF)

    print(f'Residual Analysis - {label}')
    if Q < chi_critical:
        print(f"The residual is white, Q: {Q}, chi_critical: {chi_critical}")
    else:
        print(
            f"The residual is NOT white, Q: {Q}, chi_critical: {chi_critical}")


# ----- 12.c. -----
# cross validation of time series
# ---- Cross Validation example (this example is for linear regression) start ----

# this statement is needed to define data splitter,
# specifify how many times you want to split data (n_splits) to cross validate.
tscv = TimeSeriesSplit(n_splits=5)


def cross_validation_avg(data):
    rmse = []
    for train_index, test_index in tscv.split(data):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]

        predictions, _ = getAvgModel(cv_train, cv_test)

        true_values = cv_test
        rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))

    print("RMSE: {}".format(np.mean(rmse)))

# only this method, you can pass the whole df, for others you need to pass df[dependent_variable]


def cross_validation_linreg(data, dep, feats):
    rmse = []
    for train_index, test_index in tscv.split(data):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]

        _, predictions = plotTestVPrediction(
            cv_train, cv_test, dep, feats, f'split-{len(rmse) + 1}')

        true_values = cv_test[dep]
        rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))

    print("RMSE: {}".format(np.mean(rmse)))

# trend = 'mul' | 'add' | None,
# damped = True | False, only provide if trend is not None
# seasonal = 'mul' | 'add' | None,
# seasonal_periods is a number


def cross_validation_holtwinters(data, trend=None, damped=False, seasonal=None, seasonal_periods=None):
    rmse = []
    for train_index, test_index in tscv.split(data):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]

        predictions = plotHoltWinters(
            cv_train, cv_test, trend, damped, seasonal, seasonal_periods, f'split-{len(rmse) + 1}')

        true_values = cv_test
        rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))

    print("RMSE: {}".format(np.mean(rmse)))

# trend = 'n' | 'c' (means constant) | 't' (means linear) | 'ct' (means constant and linear)
# arma = (na, nb)
# sarima = (na, d, nb, s)


def cross_validation_sarima(data, trend='n', arma=(0, 0), sarima=(0, 0, 0, 0)):
    rmse = []
    na, nb = arma
    for train_index, test_index in tscv.split(data):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]

        arima_model = predictARIMA(
            cv_train, na, nb, trend=trend, seasonal=sarima)
        predictions = arima_model.forecast(len(cv_test))

        true_values = cv_test
        rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))

    print("RMSE: {}".format(np.mean(rmse)))

# ---- Cross Validation example end ----

# %%
# ARMA data generation


def makeParams(l, d):
    return np.r_[1, d, [0] * (l - len(d) - 1)]


def armaProcess(T, ai, bi, wn_mean=WN_Mean, wn_var=WN_Var):
    na = len(ai)
    nb = len(bi)

    plen = np.max([na, nb]) + 1
    ar = makeParams(plen, ai)
    ma = makeParams(plen, bi)
    mean = wn_mean * (1 + np.sum(bi)) / (1 + np.sum(ai))
    mean = 0 if np.isnan(mean) else mean

    process = sm.tsa.ArmaProcess(ar, ma)
    y = process.generate_sample(T, scale=np.sqrt(wn_var)) + mean

    # print(f'Is this stationary process: {process.isstationary}')

    return process, y

# %%
# GPAC Table generation


def fi_j_k(ry, j, k):
    num = 1
    den = 1

    if k == 1:
        num = ry[j + 1]
        den = ry[j - k + 1]
    else:
        mat = []
        for i in range(k):
            mat.append([])
            for l in range(k-1):
                ind = np.absolute(j - l + i)
                mat[i].append(ry[ind])
            mat[i].append(ry[j + 1 + i])

        mat = np.array(mat)
        # print(f'numerator matrix: {mat}')
        num = np.linalg.det(mat)

        mat = []
        for i in range(k):
            mat.append([])
            for l in range(k):
                ind = np.absolute(j - l + i)
                mat[i].append(ry[ind])

        mat = np.array(mat)
        # print(f'denominator matrix: {mat}')
        den = np.linalg.det(mat)

    # if num == 0:
    #     return 0.0
    return (num / den)


def showGPAC(ry, j=J, k=K):
    table = []
    # print(ry)
    for i in range(j):
        table.append([])
        for l in range(1, k):
            val = fi_j_k(ry, i, l)
            table[i].append(val)

    table = pd.DataFrame(table, columns=list(range(1, k)))
    plt.figure(figsize=(12, 12))
    sns.heatmap(table, annot=True, cmap='magma', fmt=f'.{DEC}f')
    plt.title("GPAC Table")
    plt.show()

    return table

# %%
# Parameter prediction - 2. LM
# helper functions


def E(y, theta, na, nb):
    ai = theta[:na]
    bi = theta[na:]

    plen = np.max([na, nb]) + 1
    den = makeParams(plen, ai)  # AR params
    num = makeParams(plen, bi)  # MA params

    system = (den, num, 1)
    _, e = signal.dlsim(system, y)
    # return np.around(e.flatten(), decimals=12)
    return e.flatten()


def LM(y, na, nb):
    theta = [00.] * (na + nb)
    sse_iterations = {'sse': [], 'iteration': []}
    e = E(y, theta, na, nb)

    epselon = 0.001
    max = 50
    delta = 0.0000001
    mu = 0.01
    mu_max = 10000
    I = np.identity(na + nb)

    variance = None
    cov_theta = None

    # print(E(y, theta, na, nb))
    for i in range(max):
        # STEP 1
        X = []
        for j in range(na + nb):
            theta[j] += delta
            e_d = E(y, theta, na, nb)
            # print(f'e_d: {e_d}')
            xj = (e - e_d) / delta
            X.append(xj)
            theta[j] -= delta

        X = np.array(X)
        # the following formulas change because the vector X is already transposed in creation
        A = X @ X.T  # n x n (n = # of coefficients)
        # print(f'A: {A}')
        g = X @ e

        # STEP 2
        delta_theta = np.linalg.inv(A + (mu * I)) @ g
        theta_new = theta + delta_theta
        e_new = E(y, theta_new, na, nb)

        sse_iterations['sse'].append(SSE(e_new))
        sse_iterations['iteration'].append(i + 1)

        # STEP 3
        if SSE(e_new) < SSE(e):
            if np.linalg.norm(delta_theta) < epselon:
                theta = theta_new
                variance = SSE(e_new) / (len(y) - len(theta))
                cov_theta = variance * np.linalg.inv(A)
                break
            else:
                theta = theta_new
                mu /= 10

        while SSE(e_new) >= SSE(e):
            mu *= 10
            if mu > mu_max:
                print('Mu too big!')
                return theta, sse_iterations, None, None
            delta_theta = np.linalg.inv(A + (mu * I)) @ g
            theta_new = theta + delta_theta
            e_new = E(y, theta_new, na, nb)

        theta = theta_new
        e = e_new

    return theta, sse_iterations, cov_theta, variance


def genLabels(na, nb):
    def x(i): return f'a{i+1}'
    As = map(x, range(na))
    def x(i): return f'b{i+1}'
    Bs = map(x, range(nb))
    return list(As) + list(Bs)


def printLM(y, na, nb):
    theta, sse, ct, variance = LM(y, na, nb)
    labels = genLabels(na, nb)
    # part 1
    for i, v in enumerate(labels):
        print(f'Estimated Coefficient {v}: {rnd(theta[i])}')

    if ct is None:
        print('Covariance data is not available :(')
    else:
        # part 2
        for i, v in enumerate(labels):
            delta = 2 * np.sqrt(ct[i][i])
            lv = theta[i] - delta
            hv = theta[i] + delta
            print(f'Confidence intervals: {rnd(lv)} < {v} < {rnd(hv)}')

        # part 3
        # print(f'Covariance Matrix: {rnd(ct)}')

    # part 4
    if variance:
        print(f'Estimated Variance of Error: {rnd(variance)}')

    # part 6
    plt.plot(sse['iteration'], sse['sse'])
    if (len(sse['iteration']) <= 5):
        plt.xticks(sse['iteration'])
    plt.xlabel('# Iteration')
    plt.ylabel('SSE')
    plt.title('# Iterations vs SSE')
    plt.grid()
    plt.show()
