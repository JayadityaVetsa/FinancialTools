from flask import Flask, request, render_template, session
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
app.secret_key = 'your_secret_key'

# Custom zip filter
@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

def zip_lists(a, b, c, d):
    return zip(a, b, c, d)

app.jinja_env.filters['zip'] = zip_lists

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guides', methods=['GET'])
def guides():
    return render_template('guides.html')

def format_large_number(number):
    """Formats a large number into K, M, B, T notation."""
    if number < 1000:
        return f"{number}"
    elif number < 1_000_000:
        return f"{number / 1_000:.1f}K"
    elif number < 1_000_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number < 1_000_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    else:
        return f"{number / 1_000_000_000_000:.1f}T"

@app.route('/dcf-analysis', methods=['GET', 'POST'])
def dcf_analysis():
    if request.method == 'POST':
        symbol = request.form['ticker'].upper().strip()
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

            US_GDP_Growth = 0.025

            EVEBITDATV = (EV_EBITDA) * ebitda_forecast[4]
            perpuity_growth_terminal_value = (freecashflow_forecast[4]*(1+US_GDP_Growth)) / (WACC-US_GDP_Growth)

            terminal_value = (perpuity_growth_terminal_value +EVEBITDATV )/2

            PV_TV = terminal_value * discount_factor_list[4]
            enterprise_value = sum(PV_FCF_list) + PV_TV
            equity_value = enterprise_value + total_assets - total_debt
            intrinsic_value = equity_value / shares_outstanding

            #4-year price target
            enterprise_value2 = 0
            for i in range(len(freecashflow_forecast)):
                enterprise_value2 = enterprise_value2 + freecashflow_forecast[i]
            enterprise_value2 = enterprise_value2 + terminal_value
            equity_value2 = enterprise_value2 + (total_assets - total_debt)
            Fifth_intrinsic_value = equity_value2/shares_outstanding

            # Format large numbers in lists
            ebitda_forecast = [format_large_number(value) for value in ebitda_forecast]
            freecashflow_forecast = [format_large_number(value) for value in freecashflow_forecast]
            discount_factor_list = [f"{value:.4f}" for value in discount_factor_list]  # Not really large numbers, so just formatted to 4 decimal places
            PV_FCF_list = [format_large_number(value) for value in PV_FCF_list]

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
                                   WACC='{0:.2f}'.format(WACC*100),
                                   intrinsic_value='{0:.2f}'.format(intrinsic_value),
                                   earnings_growth=earnings_growth * 100,
                                   ev_ebitda_multiple='{0:.2f}'.format(EV_EBITDA),
                                   ebitda_forecast=ebitda_forecast,
                                   freecashflow_forecast=freecashflow_forecast,
                                   EVEBITDATV = format_large_number(EVEBITDATV),
                                   perpuity_growth_terminal_value = format_large_number(perpuity_growth_terminal_value),
                                   terminal_value = format_large_number(terminal_value),
                                   plot_data=plot_data,
                                   discount_factor_list = discount_factor_list,
                                   PV_FCF_list = PV_FCF_list,
                                   PV_TV = format_large_number(PV_TV),
                                   enterprise_value = format_large_number(enterprise_value),
                                   total_assets = format_large_number(total_assets),
                                   total_debt = format_large_number(total_debt),
                                   equity_value = format_large_number(equity_value),
                                   Fifth_intrinsic_value = '{0:.2f}'.format(Fifth_intrinsic_value),
                                   shares_outstanding = format_large_number(shares_outstanding),
                                   )
        except Exception as e:
            return render_template('dcf_analysis.html', error=str(e))
    return render_template('dcf_analysis.html')

@app.route('/ai-summarizer', methods=['GET', 'POST'])
def ai_summarizer():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
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
                        "Analyze the transcript provided and answer any questions the user has on these transcripts. Don't give me responses in paragraph format I want multiple lines like bulletpoints and incluse key numbers.",
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
        cleaned_response = ai_response.text.replace("*", "\n")  # Remove asterisks
        return render_template('ai_summarizer.html', question=question, response=cleaned_response)

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

@app.route('/SEC_Fillings', methods=['GET', 'POST'])
def SEC_Fillings():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        api_url = f"https://discountingcashflows.com/api/sec-filings/{ticker}/"
        response = requests.get(api_url)
        json_data = response.json()
        links_10q = []
        dates_10q = []

        links_10K = []
        dates_10k = []

        links_8k = []
        dates_8k = []


        links_4 = []
        dates_4 = []
        count = 0
        for i in range(len(json_data)):
            if json_data[i]['type'] == '4':
                links_4.append(json_data[i]['finalLink'])
                dates_4.append(json_data[i]['fillingDate'])
                count = count + 1
            if count == 40:
                break

        for i in range(len(json_data)):
            if json_data[i]['type'] == '10-Q':
                links_10q.append(json_data[i]['finalLink'])
                dates_10q.append(json_data[i]['fillingDate'])

            if json_data[i]['type'] == '10-K':
                links_10K.append(json_data[i]['finalLink'])
                dates_10k.append(json_data[i]['fillingDate'])


        count2 = 0
        for i in range(len(json_data)):
            if '8-K' in json_data[i]['type']:
                if '.jpg' in json_data[i]['finalLink'] or '.txt' in json_data[i]['finalLink']:
                    links_8k.append(json_data[i]['link'])
                else:
                    links_8k.append(json_data[i]['finalLink'])
                dates_8k.append(json_data[i]['fillingDate'])
                count2 = count2 + 1
            if count2 == 40:
                break

        max_length = max(len(dates_10q), len(dates_10k), len(dates_4), len(dates_8k))


        return render_template('SEC_Fillings.html',links_10q=links_10q,
                               dates_10q=dates_10q,
                               links_10K= links_10K,
                               dates_10k=dates_10k,
                               links_4=links_4,
                               dates_4=dates_4,
                               links_8k = links_8k,
                                dates_8k =  dates_8k,
                               max_length = max_length,
                                zip=zip)
    return render_template("SEC_Fillings.html")

@app.route("/stock_news", methods=['GET', 'POST'])
def stock_news():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        news_count = 30
        news_title = []
        news_url = []
        news_site = []
        news_images = []

        api_url = f"https://discountingcashflows.com/api/news/{ticker}/{news_count}/"
        response = requests.get(api_url)
        json_data = response.json()
        for item in json_data:
            news_title.append(item['title'])
            news_url.append(item['url'])
            news_site.append(item['site'])
            news_images.append(item['image'])


        # Configure the AI model
        genai.configure(api_key="AIzaSyCM0FzebXGOkU9TL18Q9yeB8FzeHmk45PM")

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "I will provide you a list of titles for stock news articles and I want you to analyze what is happening today and make a 1 sentence quick headline of the stock news. Keep it short",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Got it! I will analyze the titles in the list and make a headline that is short.\n",
                    ],
                },
            ]
        )
        all_news_titles = " ".join([title for title in news_title])
        ai_response = chat_session.send_message(all_news_titles+", make only one headline out of these titles")
        cleaned_response = ai_response.text.replace("**", "")
        return render_template("stock_news.html", response=cleaned_response, news_title=news_title, news_url=news_url, news_site=news_site, news_images=news_images)
    return render_template("stock_news.html")

if __name__ == '__main__':
    app.run(debug=True)
