import os
import logging
import json
import io
import time
from datetime import datetime

import azure.functions as func
import yfinance as yf
import pandas as pd
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

import pickle
import numpy as np

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

def screening_conditions(
    market_cap, price,
    drop_rate, reversal_rate, reversal_strength,
    ema20, ema50
):
    if market_cap is None or market_cap < 300:
        return False
    if price is None or price < 300:
        return False

    if drop_rate is None or drop_rate > -10:
        return False
    if reversal_rate is None or reversal_rate < 4:
        return False

    if reversal_strength is None or reversal_strength >= 1:
        return False

    if ema20 is None or ema50 is None or ema20 <= ema50:
        return False

    return True

def process_symbol(symbol, company_name, market, log, python_condition):
    try:
        log(f"[DOWNLOAD-START] {symbol}: downloading 180d/1d data")

        df = yf.download(symbol, period="180d", interval="1d")

        if df is None or df.empty:
            log(f"[DOWNLOAD-WARN] {symbol}: no daily data returned")
            return None

        log(f"[DOWNLOAD-END] {symbol}: {len(df)} rows downloaded")

        # --- market cap ---
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

        # --- indicators ---
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["ATR"] = calc_atr(df)
        df["vol_ma20"] = df["Volume"].rolling(window=20).mean()

        # --- ★ slope による反転判定 ---
        if len(df) < 25:
            log(f"[SKIP] {symbol}: insufficient data for slope check")
            return None

        ema20_now = df["EMA20"].iloc[-1]
        ema20_prev = df["EMA20"].iloc[-5]  # 5日前

        slope_prev = safe_float(df["EMA20"].iloc[-6] - df["EMA20"].iloc[-11])
        slope_now = safe_float(ema20_now - ema20_prev)

        is_reversal = (slope_prev < 0 and slope_now > 0)

        if not is_reversal:
            log(f"[NO-REV] {symbol}: slope_prev={slope_prev:.4f}, slope_now={slope_now:.4f}")
            return None

        # ★ 反転日は「5日前の終値日」
        reversal_date = df.index[-5].strftime("%Y-%m-%d")

        log(f"[REVERSAL] {symbol}: slope_prev={slope_prev:.4f} → slope_now={slope_now:.4f}, date={reversal_date}")

        # --- 以下は元のコードを維持 ---
        latest = df.iloc[-1]

        close_price = safe_float(latest["Close"])
        ema20 = safe_float(latest["EMA20"])
        ema50 = safe_float(latest["EMA50"])
        ema200 = safe_float(latest["EMA200"])
        atr = safe_float(latest["ATR"])

        vol_ma20 = safe_float(latest["vol_ma20"])
        volume = safe_float(latest["Volume"])
        volume_ratio = volume / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 0

        recent = df.tail(120)
        peak_price = safe_float(recent["High"].max())
        bottom_price = safe_float(recent["Low"].min())

        drop_rate = safe_float((bottom_price / peak_price - 1) * 100) if peak_price else None
        reversal_rate = safe_float((close_price / bottom_price - 1) * 100) if bottom_price else None

        if drop_rate and drop_rate != 0:
            reversal_strength = safe_float(reversal_rate / abs(drop_rate))
        else:
            reversal_strength = None

        # --- ★ python_condition を完全に無効化 ---
        # （反転抽出だけを行うため）
        # if python_condition and not eval_python_condition(python_condition, context):
        #     log(f"[FILTER] {symbol}: python_condition NG")
        #     return None

        short_score = (
            (reversal_strength or 0) * 0.4 +
            (volume_ratio or 0) * 0.2 +
            (slope_now or 0) * 0.2 +
            (drop_rate or 0) * 0.1 - 
            (atr or 0) * 0.1
        )

        mid_score = short_score

        gpt = gpt_score(
            symbol, company_name, close_price, market_cap,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50, slope_now,
            atr, volume, vol_ma20, volume_ratio
        )

        return {
            "symbol": symbol,
            "company_name": company_name,
            "market": market,
            "close": close_price,

            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "ATR": atr,
            "drop_rate": drop_rate,
            "reversal_rate": reversal_rate,
            "reversal_strength": reversal_strength,
            "market_cap": market_cap,
            "slope_ema20": slope_now,
            "volume_ratio": volume_ratio,

            "reversal_date": reversal_date,  # ★ UI 表示用

            "short_score": short_score,
            "mid_score": mid_score,

            "gpt_score": gpt.get("score"),
            "gpt_judgement": gpt.get("judgement"),
            "gpt_comment": gpt.get("comment"),
            "passed_python_condition": True
        }

    except Exception as e:
        log(f"[ERROR] {symbol} processing error: {e}")
        return None

def load_latest_model_from_blob():
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connect_str)
    container = blob_service.get_container_client("models")

    blobs = list(container.list_blobs())
    model_files = [b.name for b in blobs if b.name.startswith("model_")]

    if not model_files:
        raise Exception("Blob にモデルがありません")

    latest = sorted(model_files)[-1]

    downloader = container.download_blob(latest)
    model_bytes = downloader.readall()

    return pickle.loads(model_bytes)

# -----------------------------
# 特徴量生成（train_model.py と同じ）
# -----------------------------
def calc_features(df):
    df["return"] = df["Close"].pct_change()
    df["vol_change"] = df["Volume"].pct_change()
    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema20_slope"] = df["ema20"] - df["ema20"].shift(5)
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()

    features = {
        "ret_1d": df["return"].iloc[-1],
        "ret_3d": df["return"].iloc[-3:-1].mean(),
        "ret_5d": df["return"].iloc[-5:-1].mean(),
        "vol_1d": df["vol_change"].iloc[-1],
        "vol_5d": df["vol_change"].iloc[-5:-1].mean(),
        "ema20_slope": df["ema20_slope"].iloc[-1],
        "atr": df["atr"].iloc[-1],
    }
    return features

def build_features_for_symbol(symbol):
    df = yf.download(symbol, period="180d", interval="1d")
    if len(df) < 30:
        return None
    feats = calc_features(df)
    return np.array(list(feats.values())).reshape(1, -1)


# =========================
# メイン関数（screening）
# =========================
@app.function_name(name="screening")
@app.route(route="screening", methods=["POST"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening start")

    logs = []

    try:
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        result_container = os.getenv("RESULT_CONTAINER")

        today = datetime.now().strftime("%Y-%m-%d")
        log_blob_name = f"logs/{today}/screening.log"

        log_blob = blob_service.get_blob_client(
            container=result_container,
            blob=log_blob_name
        )

        try:
            log_blob.upload_blob("", overwrite=True)
            logging.info("[LOG] previous log cleared")
        except Exception as e:
            logging.error(f"[LOG-ERROR] failed to clear log: {e}")

        def log(msg):
            logs.append(msg)
            logging.info(msg)

            try:
                try:
                    old = log_blob.download_blob().readall().decode("utf-8")
                except:
                    old = ""

                new_text = old + msg + "\n"
                log_blob.upload_blob(new_text, overwrite=True)

            except Exception as e:
                logging.error(f"[LOG-ERROR] Failed to write log to blob: {e}")

        blob_csv_name = req.headers.get("X-Blob-Filename")

        if blob_csv_name:
            blob_container = "block-data"
            log(f"[BLOB] loading CSV from blob: {blob_csv_name}")

            blob_client = blob_service.get_blob_client(
                container=blob_container,
                blob=blob_csv_name
            )

            csv_text = blob_client.download_blob().readall().decode("utf-8")
            csv_filename = blob_csv_name
        else:
            csv_filename = req.headers.get("X-Filename", "uploaded.csv")
            csv_text = req.get_body().decode("utf-8")

        json_filename = csv_filename.replace(".csv", ".json")

        df_csv = pd.read_csv(io.StringIO(csv_text))

        required_cols = ["コード", "銘柄名", "市場"]
        for col in required_cols:
            if col not in df_csv.columns:
                return func.HttpResponse(
                    json.dumps({"error": f"CSV に '{col}' 列がありません"}),
                    mimetype="application/json",
                    status_code=400
                )

        name_dict = dict(zip(df_csv["コード"], df_csv["銘柄名"]))
        market_dict = dict(zip(df_csv["コード"], df_csv["市場"]))

        default_python_condition = (
            "drop_from_high_pct < -20 and "
            "rebound_from_low_pct > 25 and "
            "ema20_vs_ema50 > 5.0 and "
            "ema50_vs_ema200 > 10.0 and "
            "price_vs_ema20_pct > 2 and "
            "vol_vs_ma20 > 1.0 and "
            "atr_ratio > 1"
        )

        ui_condition = req.headers.get("X-Python-Condition")
        python_condition = ui_condition if ui_condition else default_python_condition

        results = []

        for code in df_csv["コード"]:
            time.sleep(0.3)

            symbol = f"{code}.T"
            company_name = name_dict.get(code, "不明")
            market = market_dict.get(code, "不明")

            result = process_symbol(symbol, company_name, market, log, python_condition)

            if result is not None:
                results.append(result)

        output_blob_name = f"{today}/{json_filename}"
        output_blob = blob_service.get_blob_client(result_container, output_blob_name)

        json_text = json.dumps(results, ensure_ascii=False, indent=2)
        output_blob.upload_blob(json_text, overwrite=True)

        log(f"[BLOB] JSON saved to {output_blob_name}")

        try:
            search_endpoint = os.getenv("SEARCH_ENDPOINT")
            search_key = os.getenv("SEARCH_KEY")
            index_name = "screening-results"

            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(search_key)
            )

            docs = []
            for r in results:
                doc = {
                    "id": f"{today}_{r.get('symbol').replace('.', '_')}",
                    "date": today,
                    "symbol": r.get("symbol"),
                    "company_name": r.get("company_name"),
                    "json_text": json.dumps(r, ensure_ascii=False),
                    "gpt_comment": r.get("gpt_comment"),
                    "indicators": (
                        f"drop_rate:{r.get('drop_rate')}, "
                        f"reversal_strength:{r.get('reversal_strength')}, "
                        f"volume_ratio:{r.get('volume_ratio')}, "
                        f"atr:{r.get('ATR')}"
                    )
                }
                docs.append(doc)

            if docs:
                search_client.upload_documents(documents=docs)
                log(f"[SEARCH] {len(docs)} 件を Azure AI Search に登録しました")
            else:
                log("[SEARCH] 登録対象の銘柄がありません")

        except Exception as e:
            log(f"[SEARCH-ERROR] Search 登録に失敗: {e}")

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

# =========================
# 2次スクリーニング API
# =========================
@app.function_name(name="second_screening")
@app.route(route="second_screening", methods=["POST"], auth_level="anonymous")
def second_screening(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        results = body.get("results", [])

        filtered = []
        for r in results:
            if (
                (r.get("reversal_strength") or 0) > 0.8 and
                (r.get("slope_ema20") or 0) > 30 and
                (r.get("volume_ratio") or 0) > 1.0
            ):
                filtered.append(r)

        return func.HttpResponse(
            json.dumps({
                "second_screening": filtered,
                "count": len(filtered)
            }, ensure_ascii=False, indent=2),
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("second_screening error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

# =========================
# 銘柄説明 API（RAG）
# =========================
@app.function_name(name="explain_symbol")
@app.route(route="explain_symbol", methods=["GET"], auth_level="anonymous")
def explain_symbol(req: func.HttpRequest) -> func.HttpResponse:
    try:
        symbol = req.params.get("symbol")
        if not symbol:
            return func.HttpResponse(
                json.dumps({"error": "symbol を指定してください"}),
                mimetype="application/json",
                status_code=400
            )

        search_endpoint = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = "screening-results"

        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )

        results = search_client.search(
            search_text=symbol,
            filter=f"symbol eq '{symbol}'",
            order_by=["date desc"],
            top=1
        )

        docs = list(results)
        if not docs:
            return func.HttpResponse(
                json.dumps({"error": f"{symbol} のデータが見つかりません"}),
                mimetype="application/json",
                status_code=404
            )

        doc = docs[0]
        r = json.loads(doc["json_text"])

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        prompt = f"""
あなたは短期トレードの専門家です。
以下の銘柄データをもとに、投資家向けにわかりやすく説明してください。

【銘柄】
コード: {r.get("symbol")}
企業名: {r.get("company_name")}
株価: {r.get("close")} 円
市場: {r.get("market")}

【テクニカル指標】
下落率: {r.get("drop_rate"):.2f}%
反転率: {r.get("reversal_rate"):.2f}%
反転強度: {r.get("reversal_strength"):.2f}
EMA20: {r.get("EMA20")}
EMA50: {r.get("EMA50")}
ATR: {r.get("ATR")}
出来高急増率: {r.get("volume_ratio"):.2f}

【GPT スコア】
スコア: {r.get("gpt_score")}
判断: {r.get("gpt_judgement")}

【出力形式】
- 200〜300文字の解説
- 今の相場状況で注目すべきポイント
- リスク要因（100文字以内）
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        explanation = res.choices[0].message.content.strip()

        return func.HttpResponse(
            json.dumps({
                "symbol": symbol,
                "company": r.get("company_name"),
                "date": doc["date"],
                "explanation": explanation
            }, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("explain_symbol error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


# =========================
# 3次スクリーニング（企業業績 × AI 分析版）
# =========================
@app.function_name(name="third_screening")
@app.route(route="third_screening", methods=["POST"], auth_level="anonymous")
def third_screening(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        symbols = body.get("symbols", [])

        if not symbols:
            return func.HttpResponse(
                json.dumps({"error": "symbols が空です"}),
                status_code=400
            )

        results = []

        for sym in symbols:
            ticker = yf.Ticker(sym)
            info = ticker.info

            fundamentals = {
                "売上高": info.get("totalRevenue"),
                "営業利益率": info.get("operatingMargins"),
                "純利益率": info.get("profitMargins"),
                "EPS": info.get("trailingEps"),
                "PER": info.get("trailingPE"),
                "PBR": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "売上成長率": info.get("revenueGrowth"),
                "利益成長率": info.get("earningsGrowth"),
                "フリーCF": info.get("freeCashflow"),
                "負債総額": info.get("totalDebt"),
                "現金": info.get("totalCash"),
            }

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )

            prompt = f"""
あなたはプロの株式アナリストです。
以下の企業業績データをもとに、企業の強み・弱み・リスク・総合評価を簡潔に説明してください。

銘柄: {sym}
業績データ:
{fundamentals}

日本語で、投資家向けに分かりやすく説明してください。
"""

            ai_res = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            analysis = ai_res.choices[0].message.content.strip()

            results.append({
                "symbol": sym,
                "fundamentals": fundamentals,
                "analysis": analysis
            })

        return func.HttpResponse(
            json.dumps({"results": results}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("third_screening error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500
        )

@app.function_name(name="predict_from_symbols")
@app.route(route="predict_from_symbols", methods=["POST"])
def predict_from_symbols(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        symbols = body.get("symbols", [])

        if not symbols:
            return func.HttpResponse(
                json.dumps({"error": "symbols が空です"}),
                status_code=400,
                mimetype="application/json"
            )

        model = load_latest_model_from_blob()

        results = []

        for sym in symbols:
            feats = build_features_for_symbol(sym)
            if feats is None:
                results.append({
                    "symbol": sym,
                    "prob": None
                })
                continue

            prob = float(model.predict(feats)[0] * 100)

            results.append({
                "symbol": sym,
                "prob": round(prob, 2)
            })

        return func.HttpResponse(
            json.dumps({"predictions": results}, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
