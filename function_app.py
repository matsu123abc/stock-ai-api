import azure.functions as func
import logging
import json
import os
import io
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
import ta
from openai import OpenAI
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()
client = OpenAI()


@app.function_name(name="screening")
@app.route(route="screening", methods=["GET"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening start")

    file_name = req.params.get("file")
    if not file_name:
        return func.HttpResponse(
            json.dumps({"error": "file パラメータが必要です"}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json",
        )

    try:
        conn_str = os.environ["BLOB_CONN_STR"]
        container = os.environ["BLOB_CONTAINER"]

        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service.get_blob_client(container=container, blob=file_name)

        csv_bytes = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        logging.exception("Blob 読み込み失敗")
        return func.HttpResponse(
            json.dumps({"error": f"Blob 読み込み失敗: {str(e)}"}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )

    try:
        df_result, html_str, excel_bytes = run_screening_api(df)
    except Exception as e:
        logging.exception("スクリーニング処理失敗")
        return func.HttpResponse(
            json.dumps({"error": f"スクリーニング処理失敗: {str(e)}"}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )

    if df_result is None or len(df_result) == 0:
        return func.HttpResponse(
            json.dumps({"status": "no_results", "count": 0}, ensure_ascii=False),
            mimetype="application/json",
        )

    try:
        result_container = os.environ["BLOB_RESULT_CONTAINER"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        html_blob_name = f"screening/{timestamp}.html"
        excel_blob_name = f"screening/{timestamp}.xlsx"

        blob_client_html = blob_service.get_blob_client(result_container, html_blob_name)
        blob_client_html.upload_blob(html_str, overwrite=True)

        blob_client_excel = blob_service.get_blob_client(result_container, excel_blob_name)
        blob_client_excel.upload_blob(excel_bytes, overwrite=True)

        html_url = blob_client_html.url
        excel_url = blob_client_excel.url

    except Exception as e:
        logging.exception("結果保存失敗")
        return func.HttpResponse(
            json.dumps({"error": f"結果保存失敗: {str(e)}"}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps(
            {
                "status": "ok",
                "count": int(len(df_result)),
                "html_url": html_url,
                "excel_url": excel_url,
            },
            ensure_ascii=False,
        ),
        mimetype="application/json",
    )


@app.function_name(name="monitoring")
@app.route(route="monitoring", methods=["GET"], auth_level="anonymous")
def monitoring(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("monitoring start")
    result = {"status": "ok", "message": "monitoring done"}
    return func.HttpResponse(json.dumps(result), mimetype="application/json")


def fetch_ohlcv(symbol, period="60d", interval="1h"):
    for _ in range(3):
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            logging.warning(f"{symbol} fetch error: {e}")
        time.sleep(1)
    return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df["ema20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["vol_ma20"] = df["Volume"].rolling(window=20).mean()

    return df


def screening_conditions(df, market_cap, price, symbol, name,
                         high_low_reversal_ok, reversal_strength):
    if market_cap < 300:
        return False
    if price < 300:
        return False
    if not high_low_reversal_ok:
        return False
    if reversal_strength >= 1:
        return False
    if df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]:
        return False
    return True


def gpt_score(symbol, name, price, market_cap,
              drop_rate, reversal_rate, reversal_strength,
              ema20, ema50, slope_ema20,
              atr, volume, vol_ma20, volume_ratio):

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
            model="keiba-gpt4omini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_text = raw[json_start:json_end]

        return json.loads(json_text)

    except Exception as e:
        logging.warning(f"GPT error: {e}")
        return {"score": 0, "judgement": "エラー", "comment": "GPTエラー"}


def calc_score(drop_rate, reversal_rate, reversal_strength,
               ema20, ema50, slope_ema20,
               volume_ratio, atr):

    score = 0

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

    if ema20 > ema50:
        score += 5

    if slope_ema20 > 0:
        score += 2
    if slope_ema20 > 0.5:
        score += 4

    if volume_ratio >= 2:
        score += 5
    elif volume_ratio >= 1:
        score += 3

    if atr < 20:
        score += 5
    elif atr < 30:
        score += 3
    elif atr < 40:
        score += 1

    return score


def process_symbol(row):
    symbol = row["symbol"]
    name = row["銘柄名"]

    mc = None
    try:
        ticker = yf.Ticker(symbol)
        fi = getattr(ticker, "fast_info", None)
        if fi is not None:
            mc = fi.get("market_cap", None)

        if mc is None:
            info = ticker.info
            mc = info.get("marketCap", None)

    except Exception as e:
        logging.warning(f"{symbol} market cap fetch error: {e}")
        mc = None

    if mc is None:
        return None

    market_cap = int(mc / 100000000)

    df = fetch_ohlcv(symbol)
    if df is None or df.empty:
        return None

    df = add_indicators(df)
    valid_df = df[df["Volume"] > 0]
    if len(valid_df) == 0:
        return None

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    latest = valid_df.iloc[-1]
    price = int(latest["Close"])

    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    atr = latest["atr"]

    recent = df.tail(120)

    peak_idx = recent["High"].idxmax()
    peak_price = recent.loc[peak_idx, "High"]

    bottom_idx = recent["Low"].idxmin()
    bottom_price = recent.loc[bottom_idx, "Low"]

    latest_price = recent["Close"].iloc[-1]

    drop_rate = (bottom_price / peak_price - 1) * 100
    reversal_rate = (latest_price / bottom_price - 1) * 100

    reversal_ok = reversal_rate >= 4
    drop_ok = drop_rate <= -10
    high_low_reversal_ok = drop_ok and reversal_ok

    if drop_rate != 0:
        reversal_strength = reversal_rate / abs(drop_rate)
    else:
        reversal_strength = 0

    try:
        bottom_date = bottom_idx.tz_localize(None)
    except Exception:
        bottom_date = pd.to_datetime(bottom_idx)

    vol_ma20 = latest["vol_ma20"]
    if pd.notna(vol_ma20) and vol_ma20 > 0:
        volume_ratio = latest["Volume"] / vol_ma20
    else:
        volume_ratio = 0

    if not screening_conditions(df, market_cap, price, symbol, name,
                                high_low_reversal_ok, reversal_strength):
        return None

    slope_ema20 = ema20 - df["ema20"].iloc[-5]

    score = calc_score(
        drop_rate,
        reversal_rate,
        reversal_strength,
        ema20,
        ema50,
        slope_ema20,
        volume_ratio,
        atr,
    )

    ema_condition = "○" if ema20 > ema50 else "×"

    score_info = gpt_score(
        symbol, name, price, market_cap,
        drop_rate, reversal_rate, reversal_strength,
        ema20, ema50, slope_ema20,
        atr,
        latest["Volume"], vol_ma20, volume_ratio,
    )

    judge = score_info.get("judgement", "不明")
    comment = score_info.get("comment", "")

    return {
        "symbol": symbol,
        "銘柄名": name,
        "株価": price,
        "下落率": drop_rate,
        "反転率": reversal_rate,
        "反転強度": reversal_strength,
        "底値日": bottom_date,
        "EMA20>EMA50": ema_condition,
        "ATR": atr,
        "売買判断": judge,
        "スコア": score,
        "時価総額(億円)": market_cap,
        "AIコメント": comment,
    }


def run_screening_api(jpx_df: pd.DataFrame):
    jpx_df = jpx_df.loc[:, ~jpx_df.columns.str.contains("Unnamed")]
    jpx_df["symbol"] = jpx_df["コード"].astype(str) + ".T"

    results = []

    for _, row in jpx_df.iterrows():
        result = process_symbol(row)
        if result is not None:
            results.append(result)

    if len(results) == 0:
        return None, None, None

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("反転強度", ascending=True)
    df_result = df_result.reset_index(drop=True)

    df_main = df_result.drop(columns=["AIコメント"])
    df_comment = df_result[["symbol", "銘柄名", "AIコメント"]]

    combined_html = (
        "<h2>スクリーニング結果</h2>"
        + df_main.to_html(escape=False)
        + "<hr><h2>AIコメント一覧</h2>"
        + df_comment.to_html(escape=False)
    )

    excel_buffer = io.BytesIO()
    df_result.to_excel(excel_buffer, index=False)
    excel_bytes = excel_buffer.getvalue()

    return df_result, combined_html, excel_bytes
