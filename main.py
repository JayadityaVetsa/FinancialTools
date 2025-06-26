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
import traceback

# Set the matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

ALPHA_VANTAGE_API_KEY = "TQ5LMSAFLYYILGDU"
GEMINI_API_KEY = "AIzaSyCM0FzebXGOkU9TL18Q9yeB8FzeHmk45PM" # Your Gemini Key

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
        # apikey_discountingcashflow = "030411bc-30f9-45a7-9c87-8cbd1473d592"
        try:
            # Get Free Cash Flow
            url_fcf = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=annual&apikey={apikey_fmp}'
            response_cashflow = requests.get(url_fcf)
            json_cf = response_cashflow.json()
            freecashflow_forecast = [json_cf[0]['freeCashFlow']]

            # Get Shares Outstanding from FMP Company Share Float API
            url_float = f'https://financialmodelingprep.com/stable/shares-float?symbol={symbol}&apikey={apikey_fmp}'
            response_float = requests.get(url_float)
            json_data = response_float.json()
            shares_outstanding = json_data[0]['outstandingShares']

            # Get Current Price
            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apikey_fmp}'
            response = requests.get(url)
            data = response.json()

            # Get the most recent closing price
            if 'historical' not in data or not data['historical']:
                return render_template('dcf_analysis.html', error="Could not fetch historical price data. Please try again later.")
            price = data['historical'][0]['close']
            market_cap = price * shares_outstanding

            # Get Treasury Rate using FMP Treasury Rates API
            treasury_api_url = f'https://financialmodelingprep.com/stable/treasury-rates?apikey={apikey_fmp}'
            response_treasury = requests.get(treasury_api_url)
            treasury_data = response_treasury.json()
            if not treasury_data or 'year10' not in treasury_data[0]:
                return render_template('dcf_analysis.html', error="Could not fetch 10-year treasury rate data. Please try again later.")
            Ten_treasury = treasury_data[0]['year10'] / 100

            # Financial Data
            url_balance_sheet = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={apikey_fmp}'
            response_bs = requests.get(url_balance_sheet)
            json_bs = response_bs.json()
            if not json_bs or 'totalDebt' not in json_bs[0] or 'totalCurrentAssets' not in json_bs[0]:
                return render_template('dcf_analysis.html', error="Could not fetch balance sheet data. Please try again later.")
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
            if not json_is or 'ebitda' not in json_is[0]:
                return render_template('dcf_analysis.html', error="Could not fetch income statement data. Please try again later.")
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
            intrinsic_value = round(equity_value / shares_outstanding, 2)

            #4-year price target
            enterprise_value2 = 0
            for i in range(len(freecashflow_forecast)):
                enterprise_value2 = enterprise_value2 + freecashflow_forecast[i]
            enterprise_value2 = enterprise_value2 + terminal_value
            equity_value2 = enterprise_value2 + (total_assets - total_debt)
            Fifth_intrinsic_value = round(equity_value2 / shares_outstanding, 2)

            # Format large numbers in lists
            ebitda_forecast = [format_large_number(value) for value in ebitda_forecast]
            freecashflow_forecast = [format_large_number(value) for value in freecashflow_forecast]
            discount_factor_list = [f"{value:.4f}" for value in discount_factor_list]  # Not really large numbers, so just formatted to 4 decimal places
            PV_FCF_list = [format_large_number(value) for value in PV_FCF_list]

            # Plotting using FMP API
            end_date = datetime.today()
            start_date = end_date - timedelta(days=4*365)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            price_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apikey_fmp}&from={start_str}&to={end_str}'
            price_response = requests.get(price_url)
            price_json = price_response.json()

            if 'historical' not in price_json or not price_json['historical']:
                return render_template('dcf_analysis.html', error="Could not fetch historical stock price data for plotting. Please try again later.")

            # Convert to DataFrame for plotting
            price_df = pd.DataFrame(price_json['historical'])
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df = price_df.sort_values('date')
            latest_close_price = price_df['close'].iloc[-1]

            difference = latest_close_price - intrinsic_value
            percentage_difference = (difference / intrinsic_value) * 100
            status = "over" if difference > 0 else "under"
            color = 'red' if status == "over" else 'green'

            plt.figure(figsize=(12, 6))
            plt.plot(price_df['date'], price_df['close'], label=f'Close Price\n({latest_close_price:.2f})')
            plt.axhline(y=intrinsic_value, color='r', linestyle='--', label=f' Intrinsic Value ${intrinsic_value}')
            plt.fill_between(price_df['date'], price_df['close'], intrinsic_value, where=(price_df['close'] > intrinsic_value), facecolor='red', alpha=0.3, interpolate=True, label='Over Intrinsic Value')
            plt.fill_between(price_df['date'], price_df['close'], intrinsic_value, where=(price_df['close'] < intrinsic_value), facecolor='green', alpha=0.3, interpolate=True, label='Under Intrinsic Value')
            plt.text(price_df['date'].iloc[-1], intrinsic_value, f'${difference:.2f} ({percentage_difference:.2f}%) {status}', verticalalignment='bottom',  color=color)
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
            return render_template('dcf_analysis.html', error=f"{str(e)}\n{traceback.format_exc()}")
    return render_template('dcf_analysis.html')

@app.route('/ai-summarizer', methods=['GET', 'POST'])
def ai_summarizer():
    try:
        if request.method == 'POST':
            ticker = request.form.get('ticker', '').upper().strip()
            quarter = request.form.get('quarter', '').strip()
            year = request.form.get('year', '').strip()
            question = request.form.get('question', '').strip()
            NINJAS_API_KEY = "S3IAB99btNOcaBvbQjf+Ig==6ihdvVoaAO5uCtbn"

            form_data = {'ticker': ticker, 'quarter': quarter, 'year': year, 'question': question}

            if not all([ticker, quarter, year, question]):
                return render_template('ai_summarizer.html', error="All fields are required.", **form_data)

            api_url = f"https://api.api-ninjas.com/v1/earningstranscript?ticker={ticker}&year={year}&quarter={quarter}"
            resp = requests.get(api_url, headers={'X-Api-Key': NINJAS_API_KEY})

            if resp.status_code != 200:
                error_msg = f"API Error: Could not fetch transcript for {ticker} (Q{quarter} {year}). Status: {resp.status_code}. Details: {resp.text}"
                return render_template('ai_summarizer.html', error=error_msg, **form_data)

            json_data = resp.json()
            if not json_data or 'transcript' not in json_data or not json_data['transcript']:
                error_msg = f"No transcript found for {ticker} for Q{quarter} {year}. Please verify the details and try again."
                return render_template('ai_summarizer.html', error=error_msg, **form_data)
            
            transcript = json_data['transcript']

            genai.configure(api_key="GEMINI_API_KEY")

            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                generation_config={
                    "temperature": 0.6,
                    "response_mime_type": "application/json",
                }
            )

            # ==============================================================================
            # NEW DUAL-PURPOSE PROMPT
            # This prompt commands the AI to perform two separate tasks.
            # ==============================================================================
            system_prompt = """
            You are an advanced dual-function financial analyst AI. You will perform two distinct tasks and structure your entire output as a single JSON object.

            **TASK 1: General Transcript Summary**
            - First, provide a high-level, factual summary of the entire earnings call transcript provided.
            - Identify 2-3 of the most important overall key points from the call (e.g., revenue performance, major announcements, overall tone).
            - This part should be a general overview, independent of the user's specific question.

            **TASK 2: Specific Question Analysis & Inference**
            - Second, address the user's specific question.
            - Search the transcript for a direct answer.
            - **CRITICAL RULE:** If a direct answer is NOT present, you MUST make a logical inference based on related information in the transcript (e.g., if asked about a specific product's sales and it's not mentioned, but the overall category sales are down, you can infer the product likely did not perform well).
            - You MUST clearly state that your answer is an inference and explain your reasoning. This is not optional.

            **REQUIRED JSON OUTPUT STRUCTURE:**
            Your response MUST be a valid JSON object with the following nested structure.

            {
              "general_analysis": {
                "title": "Overall Summary of the Earnings Call",
                "summary": "Your high-level summary of the entire call goes here.",
                "key_points": [
                  "The most important overall finding from the call.",
                  "Another significant theme or data point from the call."
                ]
              },
              "specific_answer": {
                "title": "Analysis of Your Question",
                "answer": "Your detailed answer to the user's specific question goes here. This can be factual or inferred.",
                "is_inferred": true,  // Set to `true` if you had to infer the answer, `false` if it was stated directly.
                "reasoning_and_disclaimer": "If `is_inferred` is true, explain your reasoning here and include a disclaimer. For example: 'The transcript does not explicitly state this, however, based on the decline in the broader 'Widgets' category, it is reasonable to infer... Note that this is an assumption and not explicitly stated.' If `is_inferred` is false, this should be null."
              }
            }
            """

            user_prompt = (
                f"USER QUESTION: \"{question}\"\n\n"
                f"EARNINGS TRANSCRIPT:\n---\n{transcript[:15000]}\n---"
            )
            
            full_prompt = [system_prompt, user_prompt]
            ai_res = model.generate_content(full_prompt)

            try:
                analysis_data = json.loads(ai_res.text)
            except (json.JSONDecodeError, TypeError) as e:
                error_msg = f"The AI model returned data in an unexpected format. This can happen with complex requests. Please try rephrasing your question. Raw AI response: {ai_res.text}"
                return render_template('ai_summarizer.html', error=error_msg, **form_data)

            return render_template('ai_summarizer.html',
                                   ai_analysis=analysis_data,
                                   **form_data)

        return render_template('ai_summarizer.html')

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return render_template('ai_summarizer.html', 
                               error=f"A critical server error occurred. Details: {str(e)}")

@app.route('/monte-carlo', methods=['GET', 'POST'])
def monte_carlo():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper().strip()
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        simulations = int(request.form['simulations'])
        days = int(request.form['days'])

        # Fetch stock data
        stock_data = mc.fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return render_template('monte_carlo.html', error="The ticker symbol you entered is incorrect. Please try again. Tickers are like this: (AAPL, MSFT, NVDA, TSLA)")

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
        ticker = request.form['ticker'].upper().strip()
        # --- NEW: Replace with your actual sec-api.io API key ---
        apikey_sec_api = "3d55f5d3d1dd31bd282d5c079c779b82b89b67fd7c9e26f0ad08a236482541f4"
        
        # --- NEW: Using sec-api.io Query API ---
        # Documentation: https://sec-api.io/docs/query-api/
        api_url = f"https://api.sec-api.io?token={apikey_sec_api}"

        # Define the form types we are interested in
        form_types_to_fetch = ["10-Q", "10-K", "8-K", "4", "6-K", "SC 13G/A"]
        
        # Construct the query for the API
        # This is more efficient as we ask the API for exactly what we need.
        # It fetches the 200 most recent filings matching the criteria.
        query = {
          "query": { "query_string": { 
              "query": f"ticker:{ticker} AND formType:(\"{'\" OR \"'.join(form_types_to_fetch)}\")" 
          }},
          "from": "0",
          "size": "200",
          "sort": [{ "filedAt": { "order": "desc" } }]
        }

        try:
            # The Query API uses a POST request with a JSON payload
            response = requests.post(api_url, json=query)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            json_data = response.json()

            # Initialize lists for each filing type
            links_10q, dates_10q = [], []
            links_10k, dates_10k = [], []
            links_8k, dates_8k = [], []
            links_4, dates_4 = [], []
            links_6k, dates_6k = [], []
            links_13g, dates_13g = [], []
            
            # --- NEW: Optimized single loop to process all filings ---
            for filing in json_data['filings']:
                form_type = filing['formType']
                # The link to the filing's landing page on sec.gov
                link = filing['linkToFilingDetails']
                # The filing date, formatted as YYYY-MM-DD
                date = filing['filedAt'].split('T')[0]

                if form_type == '10-Q':
                    links_10q.append(link)
                    dates_10q.append(date)
                elif form_type == '10-K':
                    links_10k.append(link)
                    dates_10k.append(date)
                elif form_type == '4' and len(links_4) < 40:
                    links_4.append(link)
                    dates_4.append(date)
                elif form_type == '8-K' and len(links_8k) < 40:
                    links_8k.append(link)
                    dates_8k.append(date)
                elif form_type == '6-K' and len(links_6k) < 40:
                    links_6k.append(link)
                    dates_6k.append(date)
                elif form_type == 'SC 13G/A' and len(links_13g) < 40:
                    links_13g.append(link)
                    dates_13g.append(date)
            
            # Check if any filings were found at all
            if not json_data['filings']:
                error_message = f"No SEC filings found for ticker '{ticker}'. Please check the symbol and try again. Examples: AAPL, MSFT, GOOGL."
                return render_template("SEC_Fillings.html", error=error_message)

            # Determine the maximum length of dates to maintain consistent table display
            # Note: The 'links' lists were renamed to avoid conflicts with Python built-ins
            max_length = max(len(dates_10q), len(dates_10k), len(dates_4), len(dates_8k), len(dates_6k), len(dates_13g))

            return render_template('SEC_Fillings.html', 
                                   links_10q=links_10q, dates_10q=dates_10q,
                                   links_10k=links_10k, dates_10k=dates_10k, # Renamed from links_10K to avoid case confusion
                                   links_4=links_4, dates_4=dates_4,
                                   links_8k=links_8k, dates_8k=dates_8k,
                                   links_6k=links_6k, dates_6k=dates_6k,
                                   links_13g=links_13g, dates_13g=dates_13g,
                                   max_length=max_length,
                                   zip=zip,
                                   ticker=ticker) # Pass the ticker back to display on the page

        except requests.exceptions.RequestException as e:
            # Handle potential JSON decoding errors or other request issues
            error_message = f"An error occurred while communicating with the SEC API: {str(e)}"
            return render_template('SEC_Fillings.html', error=error_message)
        except KeyError:
            # This can happen if the API response doesn't contain 'filings'
            # (e.g., if the ticker is invalid)
            error_message = f"Invalid ticker or no data found for '{ticker}'. Please enter a valid stock ticker."
            return render_template("SEC_Fillings.html", error=error_message)

    return render_template("SEC_Fillings.html")


@app.route("/stock_news", methods=['GET', 'POST'])
def stock_news():
    if request.method == 'POST':
        try:
            ticker = request.form['ticker'].upper().strip()
            api_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=50&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data or not data.get('feed'):
                error_msg = f"Could not find news for ticker '{ticker}'. It may be an invalid symbol or have no recent news."
                return render_template("stock_news.html", error=error_msg)

            # --- NEW: Using a single list of dictionaries ---
            # This is much cleaner than managing multiple lists.
            articles = []
            for article_data in data.get('feed', []):
                # We also grab the sentiment for each article now
                for ticker_sentiment in article_data.get('ticker_sentiment', []):
                    if ticker_sentiment.get('ticker') == ticker:
                        articles.append({
                            "title": article_data.get("title"),
                            "url": article_data.get("url"),
                            "site": article_data.get("source"),
                            "image": article_data.get("banner_image"),
                            "summary": article_data.get("summary"),
                            "sentiment_label": ticker_sentiment.get('ticker_sentiment_label'),
                            "sentiment_score": ticker_sentiment.get('ticker_sentiment_score')
                        })
                        break # Move to the next article once ticker is found

            if not articles:
                return render_template("stock_news.html", error=f"No news articles were found for the ticker symbol '{ticker}'.")

            # --- NEW: Advanced AI Analysis ---
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
            
            # Create a string of just the titles to send to the AI
            all_news_titles = "\n".join([article['title'] for article in articles])

            # A more sophisticated prompt asking for a JSON output
            prompt = f"""
            Analyze the following list of news headlines for the stock ticker {ticker}.
            Based on these headlines, provide a concise daily briefing.
            The output must be a valid JSON object with the following three keys: "sentiment", "summary", and "themes".
            - "sentiment": A single string describing the overall tone (e.g., "Bullish", "Bearish", "Neutral", "Mixed").
            - "summary": A 2-3 sentence paragraph summarizing the key news.
            - "themes": A list of 3-5 key topics or themes observed in the headlines (e.g., ["Earnings Report", "Analyst Ratings", "New Product"]).

            Here are the headlines:
            {all_news_titles}
            """
            
            ai_response = model.generate_content(prompt)
            
            # --- NEW: Parse the AI's JSON response safely ---
            ai_briefing = None
            try:
                # The response text might be wrapped in ```json ... ```, so we clean it up
                cleaned_text = ai_response.text.strip().replace("```json", "").replace("```", "")
                ai_briefing = json.loads(cleaned_text)
            except (json.JSONDecodeError, AttributeError):
                # If the AI fails to return valid JSON, we create a fallback
                ai_briefing = {
                    "sentiment": "Analysis Unavailable",
                    "summary": "The AI could not generate a summary for the provided news. Please review the articles manually.",
                    "themes": []
                }

            return render_template("stock_news.html",
                                   ai_briefing=ai_briefing,
                                   articles=articles,
                                   ticker=ticker)

        except requests.RequestException as e:
            error_message = f"An error occurred fetching news. The API may be unavailable. Details: {e}"
            return render_template("stock_news.html", error=error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            return render_template("stock_news.html", error=error_message)

    return render_template("stock_news.html")


if __name__ == '__main__':
    app.run(debug=True)
