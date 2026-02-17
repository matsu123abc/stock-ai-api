import os
import logging
import json
import io

import azure.functions as func
import yfinance as yf
import pandas as pd
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from datetime import datetime

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

銘柄コード: {symbol}
株価: {price}
時価総額(億円): {market_cap}

下落率: {drop_rate:.2f}%
反転率: {reversal_rate:.2f}%
反転強度: {reversal_strength:.2f}

EMA20: {ema20}
EMA50: {ema50}
EMA20の傾き: {slope_ema20:.2f}

出来高: {volume}
出来高20日平均: {vol_ma20}
出来高急増率: {volume_ratio:.2f}

ATR(14): {atr}

返答は JSON のみ。

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
# メイン関数（screening ひとつだけ）
# =========================

@app.function_name(name="screening")
@app.route(route="screening", methods=["POST"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening start")

    logs = []

    try:
        # ① Blob 接続（最優先）
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        # ② 結果保存先コンテナ
        result_container = os.getenv("RESULT_CONTAINER")

        # ③ ログ初期化（前回分を削除）
        today = datetime.now().strftime("%Y-%m-%d")
        log_blob_name = f"logs/{today}/screening.log"

        log_blob = blob_service.get_blob_client(
            container=result_container,
            blob=log_blob_name
        )

        # 空で上書き（存在しなくても OK）
        try:
            log_blob.upload_blob("", overwrite=True)
            logging.info("[LOG] previous log cleared")
        except Exception as e:
            logging.error(f"[LOG-ERROR] failed to clear log: {e}")

        # ④ log() を定義（ここから追記が始まる）
        def log(msg):
            logs.append(msg)
            logging.info(msg)

            try:
                # 既存ログを取得（なければ空）
                try:
                    old = log_blob.download_blob().readall().decode("utf-8")
                except:
                    old = ""

                # 新しいログを追記
                new_text = old + msg + "\n"
                log_blob.upload_blob(new_text, overwrite=True)

            except Exception as e:
                logging.error(f"[LOG-ERROR] Failed to write log to blob: {e}")

        # --- UI から送られた CSV のファイル名を取得 ---
        csv_filename = req.headers.get("X-Filename", "uploaded.csv")
        json_filename = csv_filename.replace(".csv", ".json")

        # --- CSV を Body から読み込む ---
        csv_text = req.get_body().decode("utf-8")
        df_csv = pd.read_csv(io.StringIO(csv_text))

        # --- 必須列チェック ---
        required_cols = ["コード", "銘柄名", "市場"]
        for col in required_cols:
            if col not in df_csv.columns:
                return func.HttpResponse(
                    json.dumps({"error": f"CSV に '{col}' 列がありません"}),
                    mimetype="application/json",
                    status_code=400
                )

        # --- 辞書化 ---
        name_dict = dict(zip(df_csv["コード"], df_csv["銘柄名"]))
        market_dict = dict(zip(df_csv["コード"], df_csv["市場"]))

        results = []

        # =========================
        # スクリーニングロジック
        # =========================
        for code in df_csv["コード"]:
            symbol = f"{code}.T"
            company_name = name_dict.get(code, "不明")
            market = market_dict.get(code, "不明")

            try:
                # --- 株価データ取得開始 ---
                log(f"[DOWNLOAD-START] {symbol}: downloading 90d/1h data")

                df = yf.download(symbol, period="90d", interval="1h")

                # --- データが空の場合 ---
                if df is None or df.empty:
                    log(f"[DOWNLOAD-WARN] {symbol}: no data returned")
                    results.append({"symbol": symbol, "error": "株価データ取得失敗"})
                    continue

                # --- 正常取得 ---
                log(f"[DOWNLOAD-END] {symbol}: {len(df)} rows downloaded")

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
                    symbol, company_name, close_price, market_cap,
                    drop_rate, reversal_rate, reversal_strength,
                    ema20, ema50, slope_ema20,
                    atr, volume, vol_ma20, volume_ratio
                )

                results.append({
                    "symbol": symbol,
                    "company_name": company_name,
                    "market": market,
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
                    "gpt_comment": gpt.get("comment")
                })

            except Exception as e:
                log(f"[DOWNLOAD-ERROR] {symbol}: Failed to download data: {e}")
                results.append({"symbol": symbol, "error": "株価データ取得エラー"})
                continue

        # --- JSON 保存 ---
        result_prefix = os.getenv("RESULT_PREFIX", "results")
        today = datetime.now().strftime("%Y-%m-%d")
        output_blob_name = f"{result_prefix}/{today}/{json_filename}"

        output_blob = blob_service.get_blob_client(result_container, output_blob_name)

        json_text = json.dumps(results, ensure_ascii=False, indent=2)
        output_blob.upload_blob(json_text, overwrite=True)

        return func.HttpResponse(
            json.dumps({
                "saved_to": output_blob_name,
                "logs": logs
            }, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("screening error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
