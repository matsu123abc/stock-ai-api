import azure.functions as func
import logging
import json
import yfinance as yf
import pandas as pd
import numpy as np

app = func.FunctionApp()

# --- 安全な float 変換 ---
def safe_float(x):
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

# --- EMA 計算 ---
def ema(series, span):
    return series.ewm(span=span).mean()

# --- ATR 計算 ---
def calc_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

@app.function_name(name="screening")
@app.route(route="screening", methods=["GET"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening step2 start")

    symbol = req.params.get("symbol")
    if not symbol:
        return func.HttpResponse(
            json.dumps({"error": "symbol が指定されていません"}),
            mimetype="application/json",
            status_code=400
        )

    try:
        df = yf.download(symbol, period="90d", interval="1h")
        if df is None or df.empty:
            return func.HttpResponse(
                json.dumps({"error": f"{symbol} の株価データが取得できませんでした"}),
                mimetype="application/json",
                status_code=500
            )

        # --- インジケータ計算 ---
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["ATR"] = calc_atr(df)

        latest = df.iloc[-1]

        # --- 反転強度（120本） ---
        recent = df.tail(120)

        peak_price = recent["High"].max()
        bottom_price = recent["Low"].min()
        latest_price = safe_float(latest["Close"])

        drop_rate = (bottom_price / peak_price - 1) * 100
        reversal_rate = (latest_price / bottom_price - 1) * 100

        if drop_rate != 0:
            reversal_strength = reversal_rate / abs(drop_rate)
        else:
            reversal_strength = 0

        result = {
            "symbol": symbol,
            "close": latest_price,
            "EMA20": safe_float(latest["EMA20"]),
            "EMA50": safe_float(latest["EMA50"]),
            "ATR": safe_float(latest["ATR"]),
            "drop_rate": drop_rate,
            "reversal_rate": reversal_rate,
            "reversal_strength": reversal_strength
        }

        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
