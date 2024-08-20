import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import io


telegram_bot_token = 'bot_token'
telegram_chat_id = 'chat_id'


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    requests.post(url, data=payload)

# Function to send an image to a Telegram bot
def send_telegram_image(image_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': telegram_chat_id}
    requests.post(url, files=files, data=data)


ticker = "AAPL"  
start_date = "2020-01-01"
end_date = "2023-01-01"


stock_data = yf.download(ticker, start=start_date, end=end_date)


print(stock_data.head())


data = stock_data[['Close']].dropna()
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)


print(data.head())

X = data[['Close']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions, squared=False)
evaluation_message = f"Root Mean Squared Error: {rmse}"
print(evaluation_message)
send_telegram_message(evaluation_message)

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="True Prices", color="blue")
plt.plot(y_test.index, predictions, label="Predicted Prices", color="red")
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plot_file = 'stock_prediction.png'
plt.savefig(plot_file)

send_telegram_image(plot_file)

plt.show()
