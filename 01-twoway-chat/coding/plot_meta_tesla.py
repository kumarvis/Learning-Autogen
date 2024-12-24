# filename: plot_meta_tesla.py
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch historical stock price data for META and TESLA
meta = yf.download('META', start='2023-01-01', end='2023-10-01')
tesla = yf.download('TSLA', start='2023-01-01', end='2023-10-01')

# Step 2: Plot the stock price change
plt.figure(figsize=(12, 6))
plt.plot(meta['Close'], label='META', color='blue')
plt.plot(tesla['Close'], label='TESLA', color='orange')
plt.title('META and TESLA Stock Price Change (2023)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid()

# Step 3: Save the chart to a PNG file
plt.savefig('meta_tesla.png')
plt.close()