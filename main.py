from flask import Flask, request, render_template
import requests
import yfinance as yf
import json
import monte_carlo as mc
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from io import BytesIO
import google.generativeai as genai

# Set the matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dcf-analysis', methods=['GET', 'POST'])
def dcf_analysis():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        apikey_fmp = 'ZyymFyFsPSWfZBrL0skr97yoYyM5Czdr'
        try:
            # Get Free Cash Flow
            url_fcf = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=annual&apikey={apikey_fmp}'
            response_cashflow = requests.get(url_fcf)
            json_cf = response_cashflow.json()
            freecashflow_forecast = [json_cf[0]['freeCashFlow']]

            # Get Shares Outstanding
            ticker = yf.Ticker(symbol)
            shares_outstanding = ticker.info.get('sharesOutstanding')

            # Get Current Price
            hist_data = ticker.history(period="1d")
            price = hist_data['Close'].iloc[0] if 'Close' in hist_data.columns and not hist_data['Close'].empty else hist_data['Open'].iloc[0]
            market_cap = shares_outstanding * price

            # Get Treasury Rate
            treasury_ticker_symbol = "^TNX"
            treasury_rate = yf.Ticker(treasury_ticker_symbol)
            hist_data = treasury_rate.history(period="1d")
            Ten_treasury = hist_data['Close'].iloc[0] / 100

            # Financial Data
            url_balance_sheet = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={apikey_fmp}'
            response_bs = requests.get(url_balance_sheet)
            json_bs = response_bs.json()
            total_debt = json_bs[0]['totalDebt']
            total_assets = json_bs[0]['totalCurrentAssets']

            # Get tax rate
            company = yf.Ticker(symbol)
            financials_json_str = company.get_financials(proxy=False).to_json()
            financials_json = json.loads(financials_json_str)
            tax_rate = next((info.get('TaxRateForCalcs') for timestamp, info in financials_json.items()), 0)

            # Get beta
            url_profile = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={apikey_fmp}'
            response_profile = requests.get(url_profile)
            json_profile = response_profile.json()
            beta = json_profile[0]['beta']

            # Get EBITDA
            url_income_statement = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={apikey_fmp}'
            response_is = requests.get(url_income_statement)
            json_is = response_is.json()
            ebitda_forecast = [json_is[0]['ebitda']]

            # WACC Calculation
            market_return = 0.08
            cost_of_equity = Ten_treasury + beta * (market_return - Ten_treasury)
            cost_of_debt = 0.05
            EDE = market_cap / (market_cap + total_debt)
            DDE = total_debt / (market_cap + total_debt)
            WACC = EDE * cost_of_equity + DDE * cost_of_debt * (1 - tax_rate)
            EV_EBITDA = market_cap / ebitda_forecast[0]

            # Forecasting
            earnings_growth = (float(request.form['growthRate']) / 100)  # User input for growth rate
            discount_factor_list = [1 / ((1 + WACC) ** (1))]
            PV_FCF_list = []

            for i in range(4):
                ebitda_forecast.append(ebitda_forecast[i] * (1 + earnings_growth))
                freecashflow_forecast.append(freecashflow_forecast[i] * (1 + earnings_growth))
                discount_factor_list.append(1 / ((1 + WACC) ** (i + 2)))

            for i in range(5):
                PV_FCF_list.append(freecashflow_forecast[i] * discount_factor_list[i])

            terminal_value = (EV_EBITDA) * ebitda_forecast[4]
            PV_TV = terminal_value * discount_factor_list[4]
            enterprise_value = sum(PV_FCF_list) + PV_TV
            equity_value = enterprise_value + total_assets - total_debt
            intrinsic_value = equity_value / shares_outstanding

            # Plotting
            end_date = datetime.today()
            start_date = end_date - timedelta(days=4*365)

            data = yf.download(symbol, start=start_date, end=end_date)
            latest_close_price = data['Close'].iloc[-1]

            difference = latest_close_price - intrinsic_value
            percentage_difference = (difference / intrinsic_value) * 100
            status = "over" if difference > 0 else "under"
            color = 'red' if status == "over" else 'green'

            plt.figure(figsize=(12, 6))
            plt.plot(data['Close'], label=f'Close Price\n({latest_close_price:.2f})')
            plt.axhline(y=intrinsic_value, color='r', linestyle='--', label=f'Target Price ${intrinsic_value}')
            plt.fill_between(data.index, data['Close'], intrinsic_value, where=(data['Close'] > intrinsic_value), facecolor='red', alpha=0.3, interpolate=True, label='Over Target Price')
            plt.fill_between(data.index, data['Close'], intrinsic_value, where=(data['Close'] < intrinsic_value), facecolor='green', alpha=0.3, interpolate=True, label='Under Target Price')
            plt.text(data.index[-1], intrinsic_value, f'${difference:.2f} ({percentage_difference:.2f}%) {status}', verticalalignment='bottom',  color=color)
            plt.title(f'{symbol} Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Save the plot to a bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf8')
            plt.close()

            return render_template('dcf_analysis.html',
                                   symbol=symbol,
                                   WACC=WACC*100,
                                   intrinsic_value=intrinsic_value,
                                   earnings_growth=earnings_growth * 100,
                                   ev_ebitda_multiple=EV_EBITDA,
                                   ebitda_forecast=ebitda_forecast,
                                   freecashflow_forecast=freecashflow_forecast,
                                   terminal_value=terminal_value,
                                   plot_data=plot_data,
                                   discount_factor_list = discount_factor_list,
                                   PV_FCF_list = PV_FCF_list,
                                   PV_TV = PV_TV,
                                   enterprise_value = enterprise_value,
                                   total_assets = total_assets,
                                   total_debt = total_debt,
                                   equity_value = equity_value,
                                   shares_outstanding = shares_outstanding,
                                   )
        except Exception as e:
            return render_template('dcf_analysis.html', error=str(e))
    return render_template('dcf_analysis.html')

@app.route('/ai-summarizer', methods=['GET', 'POST'])
def ai_summarizer():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '')
        quarter = request.form.get('quarter', '')
        year = request.form.get('year', '')
        question = request.form.get('question', '')

        if not ticker or not quarter or not year or not question:
            return render_template('ai_summarizer.html', response="All fields are required", question=question)

        api_url = f"https://discountingcashflows.com/api/transcript/{ticker}/{quarter}/{year}/"
        response = requests.get(api_url)
        json_data = response.json()
        transcript = json_data[0].get("content", "Transcript not found.")

        # Configure the AI model
        genai.configure(api_key="AIzaSyCM0FzebXGOkU9TL18Q9yeB8FzeHmk45PM")
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "Analyze the transcript provided and answer any questions the user has on these transcripts",
                        transcript,
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Got it! I have analyzed the report and I will provide answers to the user's needs\n",
                    ],
                },
            ]
        )

        ai_response = chat_session.send_message(question)
        cleaned_response = ai_response.text.replace("**", "")  # Remove asterisks
        return render_template('ai_summarizer.html', question=question, response=cleaned_response)
    return render_template('ai_summarizer.html')

@app.route('/monte-carlo', methods=['GET', 'POST'])
def monte_carlo():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        simulations = int(request.form['simulations'])
        days = int(request.form['days'])

        # Fetch stock data
        stock_data = mc.fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return render_template('monte_carlo.html', error="Failed to fetch stock data.")

        # Run Monte Carlo simulation
        price_paths = mc.monte_carlo_simulation(stock_data, simulations, days)
        if price_paths is None:
            return render_template('monte_carlo.html', error="Failed to run Monte Carlo simulation.")

        # Generate plot
        plot_data = mc.plot_simulation(ticker, price_paths, simulations, days)
        return render_template('monte_carlo.html', plot_data=plot_data)

    return render_template('monte_carlo.html')

if __name__ == '__main__':
    app.run(debug=True)
