import pandas as pd
import prophet
import matplotlib.pyplot as plt


def prophet_model(data: pd.DataFrame, col: str, params: dict, periods: int, dtype="int"):
    """시계열 예측 결과값 반환해주는 모델
    Args:
        data: 데이터프레임
        col: 피쳐 컬럼 이름
        params: prophets 파라미터
        periods: 예측 일수

    Returns:
        pd.DataFrame: 예측 피쳐 데이터프레임형태 반환
    """
    data_c = data.copy()
    # rename cols
    data_c = data_c.rename(columns={col: "y", "DATE": "ds"})

    # add params
    m = prophet.Prophet(**params)
    # m.add_country_holidays(country_name="KOR")
    m.add_seasonality(name="monthly", period=30.5, fourier_order=3, prior_scale=0.01, mode="multiplicative")

    # fit model
    m.fit(data_c)

    # future data
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # plot forecast
    fig = m.plot(forecast)
    a = prophet.plot.add_changepoints_to_plot(fig.gca(), m, forecast)

    # plot components
    m.plot_components(forecast)

    # return forecast feature
    pred = forecast["yhat"].astype(dtype)[-periods:]
    date = future.iloc[:, 0][-periods:]
    result = pd.concat([date, pred], axis=1)

    return result


params = {
    "changepoint_prior_scale": 0.035,
    "changepoint_range": 0.8,
    "seasonality_prior_scale": 5,
    "weekly_seasonality": True,
    "yearly_seasonality": False,
    "daily_seasonality": False,
    "seasonality_mode": "multiplicative",
    "holidays_prior_scale": 1,
    "interval_width": 0.8,
}

data = pd.read_csv()
pred = prophet_model(data, "ELEC_INDSUM", params=params, periods=12 * 15)
