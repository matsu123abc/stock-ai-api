import os
import logging
import json

import azure.functions as func
import yfinance as yf
import pandas as pd
from openai import AzureOpenAI

app = func.FunctionApp()

# =========================
# 共通ユーティリティ
# =========================

def safe_float(x):
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return None

def ema(series, span):
    return series.ewm(span=span).mean()

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

def calc_score(drop_rate, reversal_rate, reversal_strength,
               ema20, ema50, slope_ema20,
               volume_ratio, atr):

    drop_rate = float(drop_rate)
    reversal_rate = float(reversal_rate)
    reversal_strength = float(reversal_strength)
    ema20 = float(ema20)
    ema50 = float(ema50)
    slope_ema20 = float(slope_ema20)
    volume_ratio = float(volume_ratio)
    atr = float(atr)

    score = 0

    # ① 反転の質
    score += int(reversal_rate / 4) * 2

    if drop_rate <= -10:
        score += 3
    if drop_rate <= -15:
        score += 5

    if reversal_strength >= 0.2:
        score += 3
    if reversal_strength >= 0.4:
        score += 5
    if reversal_strength >= 0.6:
        score += 7
    if reversal_strength >= 0.8:
        score += 9

    # ② トレンド
    if ema20 > ema50:
        score += 5

    if slope_ema20 > 0:
        score += 2
    if slope_ema20 > 0.5:
        score += 4

    # ③ 出来高
    if volume_ratio >= 2:
        score += 5
    elif volume_ratio >= 1:
        score += 3

    # ④ ATR（リスク）
    if atr < 20:
        score += 5
    elif atr < 30:
        score += 3
    elif atr < 40:
        score += 1

    return score

def gpt_score(symbol, name, price, market_cap,
              drop_rate, reversal_rate, reversal_strength,
              ema20, ema50, slope_ema20,
              atr, volume, vol_ma20, volume_ratio):

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    prompt = f"""
あなたは短期トレードの専門家です。
以下の銘柄について、短期的な期待度を 0〜100 点でスコアリングし、
さらに「買い」「様子見」「避ける」のいずれかで売買判断を行ってください。

【銘柄情報】
銘柄コード: {symbol}
銘柄名: {name}
株価: {price}
時価総額(億円): {market_cap}

【反転パターン（最重要）】
高値→安値の下落率: {drop_rate:.2f}%
安値→現在の反転率: {reversal_rate:.2f}%
反転強度（反転率 ÷ 下落率の絶対値）: {reversal_strength:.2f}

【トレンド】
EMA20: {ema20}
EMA50: {ema50}
EMA20とEMA50の位置関係: {"上" if ema20 > ema50 else "下"}
EMA20の傾き（直近5本の変化量）: {slope_ema20:.2f}

【出来高】
出来高: {volume}
出来高20日平均: {vol_ma20}
出来高急増率: {volume_ratio:.2f}

【リスク】
ATR(14): {atr}

【評価方針】
あなたは短期トレードの専門家です。
この銘柄が「高値→安値→反転」という明確な反転パターンを形成し、
その後の中盤トレンドに移行しているかを中心に評価してください。

【特に重視するポイント】
- 下落率が十分に深いか（反転の前提条件）
- 反転率が明確で、反転強度が高いか
- ただし、反転強度が 1 を大きく超えた「伸び切り局面」は過熱とみなす
- EMA20 が EMA50 の上で安定して推移しているか
- 反転後の出来高が増えているか
- ATR が過度に高くなく、リスクが管理しやすいか
- 市場心理が買い優勢に傾いているか

【スコアリング方針】
- 反転の質（反転率・下落率・反転強度）を最重要視する
- 特に「反転強度が 1 未満〜1 付近の銘柄」を高く評価する
- 反転強度が 1 を大きく超える銘柄は、すでに伸び切った局面として慎重に扱う
- EMA20と出来高は補助的に評価する
- ATRはリスク調整として扱う

【コメント生成ルール】
- 200〜300文字の読み応えある分析コメントを書くこと
- 箇条書きは禁止
- 「反転の質」と「反転強度が 1 未満であること」の意味合いを中心に分析すること
- 同じ表現を繰り返さず、銘柄ごとに異なる観点で書くこと

返答は JSON のみ。前後に文章を付けないこと。

JSON形式:
{{
  "score": 数値,
  "judgement": "買い / 様子見 / 避ける",
  "comment": "200〜300文字のコメント"
}}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_text = raw[json_start:json_end]

        return json.loads(json_text)

    except Exception as e:
        return {
            "score": 0,
            "judgement": "エラー",
            "comment": f"GPTエラー: {str(e)}"
        }

# =========================
# メイン関数
# =========================

@app.function_name(name="screening")
@app.route(route="screening", methods=["GET"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening step5 start")

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
        df["vol_ma20"] = df["Volume"].rolling(window=20).mean()

        latest = df.iloc[-1]

        close_price = safe_float(latest["Close"])
        ema20 = safe_float(latest["EMA20"])
        ema50 = safe_float(latest["EMA50"])
        atr = safe_float(latest["ATR"])

        ema20_prev = safe_float(df["EMA20"].iloc[-5])
        slope_ema20 = safe_float(ema20 - ema20_prev)

        vol_ma20 = safe_float(latest["vol_ma20"])
        volume = safe_float(latest["Volume"])
        volume_ratio = volume / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 0

        # --- 反転強度 ---
        recent = df.tail(120)
        peak_price = safe_float(recent["High"].max())
        bottom_price = safe_float(recent["Low"].min())

        drop_rate = safe_float((bottom_price / peak_price - 1) * 100) if peak_price else None
        reversal_rate = safe_float((close_price / bottom_price - 1) * 100) if bottom_price else None

        if drop_rate and drop_rate != 0:
            reversal_strength = safe_float(reversal_rate / abs(drop_rate))
        else:
            reversal_strength = None

        # --- 条件判定 ---
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

        # --- スコア ---
        score = calc_score(
            drop_rate,
            reversal_rate,
            reversal_strength,
            ema20,
            ema50,
            slope_ema20,
            volume_ratio,
            atr
        )

        # --- GPT スコア ---
        gpt = gpt_score(
            symbol, symbol, close_price, market_cap,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50, slope_ema20,
            atr, volume, vol_ma20, volume_ratio
        )

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
            "slope_ema20": slope_ema20,
            "volume_ratio": volume_ratio,
            "score": score,
            "gpt_score": gpt.get("score"),
            "gpt_judgement": gpt.get("judgement"),
            "gpt_comment": gpt.get("comment"),
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
