import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import talib
import uvicorn
import logging
import json
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Custom JSON encoder to handle NaN
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Trading Platform API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for signal response with TP and SL
class TradingSignal(BaseModel):
    symbol: str
    timeframe: str
    entry_price: float
    signal_type: str
    confidence: float
    timestamp: str
    indicator: str
    stop_loss: float = None
    take_profit: float = None

# Initialize MT5 connection
def initialize_mt5():
    logger.info("Attempting to initialize MetaTrader 5 connection")
    try:
        if not mt5.initialize():
            logger.error("MetaTrader 5 initialization failed")
            raise HTTPException(status_code=500, detail="MT5 initialization failed")
        logger.info("MetaTrader 5 initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing MT5: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MT5 initialization error: {str(e)}")

# Get OHLC data from MT5
def get_ohlc_data(symbol: str, timeframe: str, count: int = 100):
    logger.debug(f"Fetching OHLC data for {symbol} on {timeframe} timeframe, count={count}")
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1
    }
    if timeframe not in timeframe_map:
        logger.warning(f"Invalid timeframe requested: {timeframe}")
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"No OHLC data retrieved for {symbol} on {timeframe}")
            raise HTTPException(status_code=500, detail=f"No data for {symbol} in {timeframe}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').astype(str)
        df['volume'] = df.get('tick_volume', df.get('volume', 0))
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in OHLC data for {symbol}: {missing_columns}")
            raise HTTPException(status_code=500, detail=f"Missing columns: {missing_columns}")
        
        logger.info(f"Successfully fetched {len(df)} OHLC records for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLC data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching OHLC data: {str(e)}")

# Calculate indicators
def calculate_indicators(
    df,
    sma_fast_period: int = 10,
    sma_slow_period: int = 20,
    bb_period: int = 20,
    bb_deviation: float = 2.0,
    rsi_period: int = 14,
    macd_fast_period: int = 12,
    macd_slow_period: int = 26,
    macd_signal_period: int = 9,
    stoch_fastk_period: int = 14,
    stoch_slowk_period: int = 3,
    stoch_slowd_period: int = 3,
    mfi_period: int = 14,
    volume_ma_period: int = 20
):
    logger.debug(f"Calculating technical indicators")
    try:
        if len(df) < max(sma_slow_period, bb_period, rsi_period, macd_slow_period, stoch_fastk_period, mfi_period, volume_ma_period):
            raise HTTPException(status_code=500, detail="Insufficient data for indicators")
        
        df['sma_fast'] = talib.SMA(df['close'], timeperiod=sma_fast_period)
        df['sma_slow'] = talib.SMA(df['close'], timeperiod=sma_slow_period)
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=bb_deviation, nbdevdn=bb_deviation)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=stoch_fastk_period, slowk_period=stoch_slowk_period, slowd_period=stoch_slowd_period)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=mfi_period)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['support1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['resistance1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['volume_ma'] = talib.SMA(df['volume'], timeperiod=volume_ma_period)
        
        logger.info("Indicators calculated successfully")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating indicators: {str(e)}")

# Prepare data for ML models
def prepare_ml_data(df):
    logger.debug("Preparing data for ML models")
    try:
        # Features: OHLC, indicators
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'mfi',
            'pivot', 'support1', 'resistance1', 'volume_ma'
        ]
        X = df[features].fillna(0)
        
        # Target: Generate labels based on price movement
        df['return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = np.where(df['return'] > 0.001, 'Buy', np.where(df['return'] < -0.001, 'Sell', 'Hold'))
        y = df['label'].iloc[:-1]  # Exclude last row (no future return)
        X = X.iloc[:-1]  # Align with y
        
        return X, y
    except Exception as e:
        logger.error(f"Error preparing ML data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preparing ML data: {str(e)}")

# Random Forest Model
class RFSignalModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X, y):
        logger.info("Training Random Forest model")
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            logger.info(f"Random Forest model trained with accuracy: {score:.4f}")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            raise

    def predict(self, X):
        if not self.is_trained:
            logger.warning("Random Forest model not trained, returning None")
            return None
        try:
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)
            prob = self.model.predict_proba(X_scaled)
            confidence = np.max(prob, axis=1)[0]
            return pred[0], confidence
        except Exception as e:
            logger.error(f"Error predicting with Random Forest model: {str(e)}")
            return None, 0.0

# DQN Model for Reinforcement Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNTrader:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.action_dim = action_dim
        self.is_trained = False

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, X, prices):
        logger.info("Training DQN model")
        try:
            for i in range(len(X) - 1):
                state = X.iloc[i].values
                next_state = X.iloc[i + 1].values
                price = prices.iloc[i]
                next_price = prices.iloc[i + 1]
                
                # Simple reward: profit/loss from action
                action = self.act(state)
                if action == 0:  # Buy
                    reward = (next_price - price) / price * 100
                elif action == 1:  # Sell
                    reward = (price - next_price) / price * 100
                else:  # Hold
                    reward = 0
                
                self.remember(state, action, reward, next_state, False)
                
                if len(self.memory) > self.batch_size:
                    self.replay()
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            
            self.is_trained = True
            logger.info("DQN model trained")
        except Exception as e:
            logger.error(f"Error training DQN model: {str(e)}")
            raise

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        if not self.is_trained:
            logger.warning("DQN model not trained, returning None")
            return None, 0.0
        action = self.act(state)
        action_map = {0: "Buy", 1: "Sell", 2: "Hold"}
        return action_map[action], 0.85  # Fixed confidence for simplicity

# Global ML models
rf_model = RFSignalModel()
dqn_model = DQNTrader(state_dim=21, action_dim=3)  # 21 features, 3 actions (Buy, Sell, Hold)

# Train ML models
def train_ml_models(symbol, timeframe):
    logger.info(f"Training ML models for {symbol} on {timeframe}")
    try:
        # Fetch historical data (more data for training)
        df = get_ohlc_data(symbol, timeframe, count=1000)
        df = calculate_indicators(df)
        X, y = prepare_ml_data(df)
        
        # Train Random Forest
        rf_model.train(X, y)
        
        # Train DQN
        dqn_model.train(X, df['close'])
        
        logger.info("ML models trained successfully")
    except Exception as e:
        logger.error(f"Error training ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training ML models: {str(e)}")

# Generate trading signals with TP, SL, and AI
def generate_signal(df, symbol: str, timeframe: str):
    logger.debug(f"Generating trading signal for {symbol} on {timeframe}")
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []

        # Risk Management Parameters
        risk_percentage = 0.01  # 1% risk per trade
        risk_reward_ratio = 2.0  # 1:2 risk-to-reward ratio

        def calculate_tp_sl(entry_price, signal_type, support, resistance):
            if signal_type == "Buy":
                sl = min(entry_price * (1 - risk_percentage), support if not np.isnan(support) else entry_price * (1 - risk_percentage))
                tp = entry_price + (entry_price - sl) * risk_reward_ratio
                tp = min(tp, resistance if not np.isnan(resistance) else tp)
            elif signal_type == "Sell":
                sl = max(entry_price * (1 + risk_percentage), resistance if not np.isnan(resistance) else entry_price * (1 + risk_percentage))
                tp = entry_price - (sl - entry_price) * risk_reward_ratio
                tp = max(tp, support if not np.isnan(support) else tp)
            else:
                sl = None
                tp = None
            return sl, tp

        # SMA Signal
        if prev['sma_fast'] < prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=0.7, timestamp=str(latest['time']), indicator="SMA",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"SMA Buy signal generated for {symbol} with SL={sl}, TP={tp}")
        elif prev['sma_fast'] > prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow']:
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=0.7, timestamp=str(latest['time']), indicator="SMA",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"SMA Sell signal generated for {symbol} with SL={sl}, TP={tp}")

        # MACD Signal
        if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=0.75, timestamp=str(latest['time']), indicator="MACD",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"MACD Buy signal generated for {symbol} with SL={sl}, TP={tp}")
        elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=0.75, timestamp=str(latest['time']), indicator="MACD",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"MACD Sell signal generated for {symbol} with SL={sl}, TP={tp}")

        # Bollinger Bands Signal
        if prev['close'] < prev['bb_upper'] and latest['close'] > latest['bb_upper']:
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=0.65, timestamp=str(latest['time']), indicator="Bollinger Bands",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Bollinger Bands Buy signal generated for {symbol} with SL={sl}, TP={tp}")
        elif prev['close'] > prev['bb_lower'] and latest['close'] < latest['bb_lower']:
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=0.65, timestamp=str(latest['time']), indicator="Bollinger Bands",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Bollinger Bands Sell signal generated for {symbol} with SL={sl}, TP={tp}")

        # Stochastic Signal
        if prev['stoch_k'] < 80 and latest['stoch_k'] > 80 and latest['stoch_k'] > latest['stoch_d']:
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=0.6, timestamp=str(latest['time']), indicator="Stochastic",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Stochastic Sell signal generated for {symbol} with SL={sl}, TP={tp}")
        elif prev['stoch_k'] > 20 and latest['stoch_k'] < 20 and latest['stoch_k'] < latest['stoch_d']:
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=0.6, timestamp=str(latest['time']), indicator="Stochastic",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Stochastic Buy signal generated for {symbol} with SL={sl}, TP={tp}")

        # MFI Signal
        if prev['mfi'] < 80 and latest['mfi'] > 80:
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=0.65, timestamp=str(latest['time']), indicator="MFI",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"MFI Sell signal generated for {symbol} with SL={sl}, TP={tp}")
        elif prev['mfi'] > 20 and latest['mfi'] < 20:
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=0.65, timestamp=str(latest['time']), indicator="MFI",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"MFI Buy signal generated for {symbol} with SL={sl}, TP={tp}")

        # Combined Signal (Support/Resistance + Volume + MFI)
        volume_surge = latest['volume'] > 1.5 * latest['volume_ma'] if not np.isnan(latest['volume_ma']) else False
        near_support = abs(latest['close'] - latest['support1']) / latest['close'] < 0.005 if not np.isnan(latest['support1']) else False
        near_resistance = abs(latest['close'] - latest['resistance1']) / latest['close'] < 0.005 if not np.isnan(latest['resistance1']) else False

        if near_support and (volume_surge or latest['volume'] == 0) and latest['mfi'] < 40:
            confidence = 0.8 if volume_surge else 0.7
            sl, tp = calculate_tp_sl(latest['close'], "Buy", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Buy", confidence=confidence, timestamp=str(latest['time']), 
                indicator="Combined (S/R + Volume + MFI)",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Combined Buy signal generated for {symbol} with SL={sl}, TP={tp}")
        elif near_resistance and (volume_surge or latest['volume'] == 0) and latest['mfi'] > 60:
            confidence = 0.8 if volume_surge else 0.7
            sl, tp = calculate_tp_sl(latest['close'], "Sell", latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type="Sell", confidence=confidence, timestamp=str(latest['time']), 
                indicator="Combined (S/R + Volume + MFI)",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Combined Sell signal generated for {symbol} with SL={sl}, TP={tp}")

        # AI Signals
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'mfi',
            'pivot', 'support1', 'resistance1', 'volume_ma'
        ]
        X_latest = pd.DataFrame([latest])[features].fillna(0)
        
        # Random Forest Signal
        rf_signal, rf_confidence = rf_model.predict(X_latest)
        if rf_signal:
            sl, tp = calculate_tp_sl(latest['close'], rf_signal, latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type=rf_signal, confidence=rf_confidence, timestamp=str(latest['time']),
                indicator="Random Forest AI",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"Random Forest AI signal generated: {rf_signal} with confidence {rf_confidence:.4f}")

        # DQN Signal
        dqn_signal, dqn_confidence = dqn_model.predict(X_latest.values[0])
        if dqn_signal:
            sl, tp = calculate_tp_sl(latest['close'], dqn_signal, latest['support1'], latest['resistance1'])
            signals.append(TradingSignal(
                symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
                signal_type=dqn_signal, confidence=dqn_confidence, timestamp=str(latest['time']),
                indicator="DQN AI",
                stop_loss=sl, take_profit=tp
            ))
            logger.info(f"DQN AI signal generated: {dqn_signal} with confidence {dqn_confidence:.4f}")

        if signals:
            strongest_signal = max(signals, key=lambda x: x.confidence)
            logger.info(f"Selected strongest signal: {strongest_signal.signal_type} from {strongest_signal.indicator}")
            return strongest_signal
        logger.info(f"No actionable signal for {symbol}, returning Hold")
        return TradingSignal(
            symbol=symbol, timeframe=timeframe, entry_price=latest['close'],
            signal_type="Hold", confidence=0.5, timestamp=str(latest['time']), indicator="None",
            stop_loss=None, take_profit=None
        )
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")

# API endpoint to get trading signal
@app.get("/signal/{symbol}/{timeframe}", response_model=TradingSignal)
async def get_trading_signal(symbol: str, timeframe: str):
    logger.info(f"Received request for trading signal: {symbol}, {timeframe}")
    try:
        initialize_mt5()
        df = get_ohlc_data(symbol, timeframe)
        df = calculate_indicators(df)
        # Train ML models if not trained
        if not rf_model.is_trained or not dqn_model.is_trained:
            train_ml_models(symbol, timeframe)
        signal = generate_signal(df, symbol, timeframe)
        logger.info(f"Trading signal generated successfully for {symbol}")
        return signal
    except Exception as e:
        logger.error(f"Error generating trading signal for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Shutting down MetaTrader 5 connection")
        mt5.shutdown()

# API endpoint to get OHLC data with indicators
@app.get("/ohlc/{symbol}/{timeframe}")
async def get_ohlc(
    symbol: str,
    timeframe: str,
    sma_fast_period: int = Query(10, ge=1, description="Period for fast SMA"),
    sma_slow_period: int = Query(20, ge=1, description="Period for slow SMA"),
    bb_period: int = Query(20, ge=1, description="Period for Bollinger Bands"),
    bb_deviation: float = Query(2.0, ge=0.1, le=5.0, description="Standard deviation for Bollinger Bands"),
    rsi_period: int = Query(14, ge=1, description="Period for RSI"),
    macd_fast_period: int = Query(12, ge=1, description="Fast period for MACD"),
    macd_slow_period: int = Query(26, ge=1, description="Slow period for MACD"),
    macd_signal_period: int = Query(9, ge=1, description="Signal period for MACD"),
    stoch_fastk_period: int = Query(14, ge=1, description="Fast K period for Stochastic"),
    stoch_slowk_period: int = Query(3, ge=1, description="Slow K period for Stochastic"),
    stoch_slowd_period: int = Query(3, ge=1, description="Slow D period for Stochastic"),
    mfi_period: int = Query(14, ge=1, description="Period for MFI"),
    volume_ma_period: int = Query(20, ge=1, description="Period for Volume MA")
):
    logger.info(f"Received request for OHLC data: {symbol}, {timeframe}")
    try:
        initialize_mt5()
        df = get_ohlc_data(symbol, timeframe)
        df = calculate_indicators(
            df, sma_fast_period, sma_slow_period, bb_period, bb_deviation, rsi_period,
            macd_fast_period, macd_slow_period, macd_signal_period, stoch_fastk_period,
            stoch_slowk_period, stoch_slowd_period, mfi_period, volume_ma_period
        )
        result = df[[
            'time', 'open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow',
            'bb_upper', 'bb_middle', 'bb_lower', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'mfi', 'pivot', 'support1', 'resistance1', 'volume_ma'
        ]].replace({np.nan: None})
        logger.info(f"OHLC data with indicators retrieved successfully for {symbol}")
        return result.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error retrieving OHLC data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Shutting down MetaTrader 5 connection")
        mt5.shutdown()

# WebSocket endpoint for real-time data
@app.websocket("/ws/{symbol}/{timeframe}")
async def websocket_endpoint(websocket: WebSocket, symbol: str, timeframe: str):
    await websocket.accept()
    logger.info(f"WebSocket connection established for {symbol} on {timeframe}")
    try:
        initialize_mt5()
        # Train ML models if not trained
        if not rf_model.is_trained or not dqn_model.is_trained:
            train_ml_models(symbol, timeframe)
        while True:
            try:
                df = get_ohlc_data(symbol, timeframe, count=1)
                df = calculate_indicators(df)
                signal = generate_signal(df, symbol, timeframe)
                data = {
                    "type": "signal",
                    "payload": signal.dict()
                }
                await websocket.send_json(data, mode="text")
                logger.debug(f"Sent signal data via WebSocket: {data}")
                
                ohlc_data = df[[
                    'time', 'open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow',
                    'bb_upper', 'bb_middle', 'bb_lower', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'stoch_k', 'stoch_d', 'mfi', 'pivot', 'support1', 'resistance1', 'volume_ma'
                ]].replace({np.nan: None}).to_dict(orient="records")[0]
                ohlc_message = {
                    "type": "ohlc",
                    "payload": ohlc_data
                }
                await websocket.send_json(ohlc_message, mode="text")
                logger.debug(f"Sent OHLC data via WebSocket: {ohlc_message}")
                
                await asyncio.sleep(60)  # Update every 60 seconds
            except Exception as e:
                logger.error(f"Error in WebSocket loop for {symbol}: {str(e)}")
                await websocket.send_json({"type": "error", "message": str(e)})
                break
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {str(e)}")
    finally:
        logger.info(f"Closing WebSocket connection for {symbol}")
        mt5.shutdown()
        await websocket.close()

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)