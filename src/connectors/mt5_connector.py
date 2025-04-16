import MetaTrader5 as mt5
from src.utils.logger import logger
from src.utils.config_loader import load_config
import os

class MT5Connector:
    def __init__(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_root, "../../configs/credentials.yaml")
        self.config = load_config(config_path)['mt5']
        self.connected = False

    def initialize(self):
        if not mt5.initialize():
            logger.error(f"❌ خطا در راه‌اندازی MT5: {mt5.last_error()}")
            return False

        authorized = mt5.login(self.config['login'], password=self.config['password'], server=self.config['server'])
        if authorized:
            logger.info("✅ اتصال موفق به متاتریدر ۵.")
            self.connected = True
            return True
        else:
            logger.error(f"❌ ورود به حساب MT5 ناموفق بود: {mt5.last_error()}")
            return False

    def shutdown(self):
        mt5.shutdown()
        self.connected = False
        logger.info("📴 ارتباط با متاتریدر بسته شد.")
