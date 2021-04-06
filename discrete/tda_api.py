import json

from tda import auth

import config

try:
    c = auth.client_from_token_file(config.token_path, config.api_key)
except FileNotFoundError:
    from selenium import webdriver

    with webdriver.Chrome(executable_path=config.chrome_driver_path) as driver:
        c = auth.client_from_login_flow(
            driver, config.api_key, config.redirect_uri, config.token_path
        )

r = c.get_price_history(
    "AAPL",
    period_type=c.Client.PriceHistory.PeriodType.YEAR,
    period=c.Client.PriceHistory.Period.TWENTY_YEARS,
    frequency_type=c.Client.PriceHistory.FrequencyType.DAILY,
    frequency=c.Client.PriceHistory.Frequency.DAILY,
)

assert r.status_code == 200, r.raise_for_status()
print(json.dumps(r.json(), indent=4))

# r = c.get_quote('AAPL')
# r = c.get_option_chain('AAPL')
r = c.search_instruments(["AAPL"],
    projection=c.Instrument.Projection.FUNDAMENTAL)
assert r.status_code == 200, r.raise_for_status()
print(json.dumps(r.json(), indent=4))
