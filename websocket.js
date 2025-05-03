const connectWebSocket = (symbol, timeframe, onDataReceived) => {
  const ws = new WebSocket(`ws://localhost:8000/ws/${symbol}/${timeframe}`);
  
  ws.onopen = () => {
    console.log(`WebSocket connected for ${symbol} ${timeframe}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onDataReceived(data);
    } catch (error) {
      console.error("Error parsing WebSocket data:", error);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  ws.onclose = () => {
    console.log("WebSocket closed, attempting to reconnect...");
    setTimeout(() => connectWebSocket(symbol, timeframe, onDataReceived), 5000);
  };

  return ws;
};

export { connectWebSocket };