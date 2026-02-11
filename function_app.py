import azure.functions as func
import logging
import json
import yfinance as yf
import pandas as pd

app = func.FunctionApp()

# --- 安全な float 変換 ---
def safe_float(x):
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return None

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
    logging.info("screening step3 start")

    symbol = req.params.get("symbol")
    if not symbol:
        return func.HttpResponse(
            json.dumps({"error": "symbol が指定されていません"}),
            mimetype="application/json",
            status_code=400
        )

    try:
        # --- 株価データ ---
        df = yf.download(symbol, period="90d", interval="1h")
        if df is None or df.empty:
            return func.HttpResponse(
                json.dumps({"error": f"{symbol} の株価データが取得できませんでした"}),
                mimetype="application/json",
                status_code=500
            )

        # --- 時価総額 ---
        try:
            ticker = yf.Ticker(symbol)
            fi = getattr(ticker, "fast_info", None)
            mc = None

            if fi is not None:
                mc = fi.get("market_cap", None)

            if mc is None:
                info = ticker.info
                mc = info.get("marketCap", None)

            market_cap = int(mc / 100000000) if mc else None

        except:
            market_cap = None

        # --- インジケータ ---
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["ATR"] = calc_atr(df)

        latest = df.iloc[-1]

        close_price = safe_float(latest["Close"])
        ema20 = safe_float(latest["EMA20"])
        ema50 = safe_float(latest["EMA50"])
        atr = safe_float(latest["ATR"])

        # --- 反転強度 ---
        recent = df.tail(120)

        peak_price = safe_float(recent["High"].max())
        bottom_price = safe_float(recent["Low"].min())

        drop_rate = (bottom_price / peak_price - 1) * 100 if peak_price else None
        reversal_rate = (close_price / bottom_price - 1) * 100 if bottom_price else None

        if drop_rate and drop_rate != 0:
            reversal_strength = reversal_rate / abs(drop_rate)
        else:
            reversal_strength = None

        # --- screening 条件 ---
        cond_drop = drop_rate is not None and drop_rate <= -10
        cond_rev = reversal_rate is not None and reversal_rate >= 4
        cond_strength = reversal_strength is not None and reversal_strength < 1
        cond_ema = ema20 is not None and ema50 is not None and ema20 > ema50
        cond_price = close_price is not None and close_price >= 300
        cond_mc = market_cap is not None and market_cap >= 300

        final_judgement = all([
            cond_drop,
            cond_rev,
            cond_strength,
            cond_ema,
            cond_price,
            cond_mc
        ])

        result = {
            "symbol": symbol,
            "close": close_price,
            "EMA20": ema20,
            "EMA50": ema50,
            "ATR": atr,
            "drop_rate": drop_rate,
            "reversal_rate": reversal_rate,
            "reversal_strength": reversal_strength,
            "market_cap": market_cap,
            "conditions": {
                "drop_ok": cond_drop,
                "reversal_ok": cond_rev,
                "strength_ok": cond_strength,
                "ema_ok": cond_ema,
                "price_ok": cond_price,
                "market_cap_ok": cond_mc,
                "final_judgement": final_judgement
            }
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
