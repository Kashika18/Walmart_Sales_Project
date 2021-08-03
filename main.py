from flask import Flask, request, render_template
import pickle
import numpy as np
import datetime
# from datetime import date
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
#model_month = pickle.load(open("model_month.pkl", "rb"))
@app.route('/')
def sales_forecast():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict_data():
    values = [str(x) for x in request.form.values()]
    val = pd.to_datetime(values, format='%Y-%m-%d')
    df = pd.DataFrame({'ds': val})
    forecast=model.predict(df)
    sales_val=forecast['yhat'].values[0]
    print(sales_val)
    return render_template('index.html', pred="Your Store Weekly Sales "+str(sales_val))

@app.route('/forecastmon', methods=["POST"])
def forecastMon():
    mon, year=[x for x in request.form.values()]

    datetime_object = datetime.datetime.strptime(mon, "%b")
    mon_no = datetime_object.month
    mon_no=int(mon_no)
    year=int(year)
    #future_month = model_month.make_future_dataframe(periods=5909)
    #forecast_month = model_month.predict(future_month)
    forecast_month['month'] = forecast_month.ds.dt.month
    forecast_month['year'] = forecast_month.ds.dt.year
    #forecast_month_sum = forecast_month.groupby(['year', 'month']).agg(month_yhat_sum = ('yhat', 'sum'),month_yhat_upper_sum = ('yhat_upper', 'sum'),month_yhat_lower_sum = ('yhat_lower', 'sum')).reset_index()
    #resuld_df = pd.Dataframe(forecast_month[['ds','month', 'year', 'yhat','yhat_upper','yhat_lower']])
    #forecast_month_sum=forecast_month_sum[forecast_month_sum['year'].isin([year]) & forecast_month_sum['month'].isin([mon_no])]
    #mon_sale=forecast_month_sum['month_yhat_sum']
    # print(mon, year)
    graph_month = plt.plot(forecast_month['ds'],forecast_month['yhat'])
    return render_template('index.html', pred1 = graph_month)
    # return render_template('index.html', pred1=mon,year)


if __name__ == '__main__' :
    app.run(debug=True)