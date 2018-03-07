from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import pandas as pd
import matplotlib.pyplot as plt


def run_model(X, tr, te, p0, p1, model_type='rf'):
    if model_type == 'rf':
        m0, m1 = RandomForestRegressor(n_estimators=200), RandomForestRegressor(n_estimators=200)
    elif model_type == 'ridge':
        m0, m1 = Ridge(), Ridge()
    elif model_type == 'lasso':
        m0, m1 = Lasso(), Lasso()

    # predict starts
    m1.fit(X.ix[tr], p1.ix[tr])
    p1_pred = pd.Series(m1.predict(X.ix[te]), index=te)

    fig = plt.figure(0)
    plt.plot(p1.ix[te], 'black', label='testing')
    plt.plot(p1_pred, 'blue', label='prediction')
    fig.legend()

    m0.fit(X.ix[tr], p0.ix[tr])
    p0_pred = pd.Series(m0.predict(X.ix[te]), index=te)

    # fig = plt.figure(1)
    # plt.plot(p0.ix[te], 'black', label='testing')
    # plt.plot(p0_pred, 'blue', label='prediction')
    # fig.legend()

    return m0, m1
