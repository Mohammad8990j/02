import MetaTrader5 as mt5
from src.utils.logger import logger
from src.connectors.mt5_connector import MT5Connector
from src.connectors.data_fetcher import DataFetcher
from src.utils.file_helper import ensure_dir
from src.utils.config_loader import load_config
import os

# تبدیل رشته تایم‌فریم به ENUM متاتریدر
def timeframe_str_to_enum(tf_str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    return mapping.get(tf_str.upper(), mt5.TIMEFRAME_M15)

def main():
    logger.info("✅ راه‌اندازی ربات تریدر پیشرفته آغاز شد.")

    # بارگذاری تنظیمات
    try:
        config = load_config()
        symbol = config["trading"]["symbol"]
        tf_str = config["trading"]["timeframe"]
        bars = config["trading"]["bars"]
        login = config["account"]["login"]
        password = config["account"]["password"]
        server = config["account"]["server"]
        timeframe = timeframe_str_to_enum(tf_str)
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری تنظیمات: {e}")
        return

    # اتصال به متاتریدر
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        logger.error("❌ اتصال به MT5 با شکست مواجه شد.")
        return

    # دریافت داده‌ها
    fetcher = DataFetcher(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        login=login,
        password=password,
        server=server
    )
    df = fetcher.fetch_data()
    if df is not None:
        save_dir = os.path.join(os.path.dirname(__file__), "data", "historical")
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{symbol}_{tf_str}.csv")
        fetcher.save_to_csv(df, save_path)

    # قطع اتصال
    mt5_conn.shutdown()


if __name__ == "__main__":
    main()
