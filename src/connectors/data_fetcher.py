import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime
from src.preprocessing.feature_engineering import calculate_indicators

class DataFetcher:
    def __init__(self, symbol, timeframe, bars, login, password, server):
        """
        Constructor for DataFetcher class.
        
        :param symbol: Symbol (e.g., "EURUSD")
        :param timeframe: Timeframe for data (e.g., mt5.TIMEFRAME_M1)
        :param bars: Number of historical bars to fetch
        :param login: MetaTrader 5 login credentials
        :param password: MetaTrader 5 password
        :param server: MetaTrader 5 server information
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self):
        """
        Connects to MetaTrader5 using provided credentials.
        """
        if not mt5.initialize():
            logging.error(f"Initialization failed: {mt5.last_error()}")
            return False

        if not mt5.login(self.login, password=self.password, server=self.server):
            logging.error(f"Login failed: {mt5.last_error()}")
            return False

        self.connected = True
        logging.info("Successfully connected to MetaTrader5.")
        return True

    def fetch_data(self):
        """
        Fetch historical data from MetaTrader5.
        
        :return: DataFrame with historical OHLC data along with calculated indicators
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader5. Please connect first.")
            return None

        # Request historical data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bars)

        # Check if data was fetched correctly
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to fetch data for {self.symbol} on timeframe {self.timeframe}.")
            return None
        
        # Convert raw data into DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calculate technical indicators
        df = calculate_indicators(df)

        return df

    def save_to_csv(self, df, path):
        """
        Saves the DataFrame to a CSV file.
        
        :param df: DataFrame containing OHLC data and indicators
        :param path: Path where the file will be saved
        """
        try:
            df.to_csv(path, index=False)
            logging.info(f"Data saved successfully to {path}.")
        except Exception as e:
            logging.error(f"Failed to save data to {path}: {str(e)}")

    def disconnect(self):
        """
        Disconnects from MetaTrader5.
        """
        if self.connected:
            mt5.shutdown()
            logging.info("Disconnected from MetaTrader5.")
            self.connected = False
