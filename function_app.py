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

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import time

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
    # --- 基本条件 ---
    if market_cap is None or market_cap < 300:
        return False
    if price is None or price < 300:
        return False

    # --- 高値→安値→反転パターン ---
    if drop_rate is None or drop_rate > -10:
        return False
    if reversal_rate is None or reversal_rate < 4:
        return False

    # --- 反転強度 < 1 ---
    if reversal_strength is None or reversal_strength >= 1:
        return False

    # --- EMA20 > EMA50 ---
    if ema20 is None or ema50 is None or ema20 <= ema50:
        return False

    return True

def process_symbol(symbol, company_name, market, log, python_condition):
    try:
        # =========================
        # ① 日足データ（180日・1d）
        # =========================
        log(f"[DOWNLOAD-START] {symbol}: downloading 180d/1d data")

        df = yf.download(symbol, period="180d", interval="1d")

        if df is None or df.empty:
            log(f"[DOWNLOAD-WARN] {symbol}: no daily data returned")
            return None

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

        # =========================
        # ② 日足インジケータ計算
        # =========================
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)   # ★ 追加
        df["ATR"] = calc_atr(df)
        df["vol_ma20"] = df["Volume"].rolling(window=20).mean()

        latest = df.iloc[-1]

        close_price = safe_float(latest["Close"])
        ema20 = safe_float(latest["EMA20"])
        ema50 = safe_float(latest["EMA50"])
        ema200 = safe_float(latest["EMA200"])
        atr = safe_float(latest["ATR"])

        # EMA20 の傾き（5日前との差）
        ema20_prev = safe_float(df["EMA20"].iloc[-5])
        slope_ema20 = safe_float(ema20 - ema20_prev)

        # 出来高
        vol_ma20 = safe_float(latest["vol_ma20"])
        volume = safe_float(latest["Volume"])
        volume_ratio = volume / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 0

        # =========================
        # ③ 反転パターン（日足ベース）
        # =========================
        recent = df.tail(120)
        peak_price = safe_float(recent["High"].max())
        bottom_price = safe_float(recent["Low"].min())

        drop_rate = safe_float((bottom_price / peak_price - 1) * 100) if peak_price else None
        reversal_rate = safe_float((close_price / bottom_price - 1) * 100) if bottom_price else None

        if drop_rate and drop_rate != 0:
            reversal_strength = safe_float(reversal_rate / abs(drop_rate))
        else:
            reversal_strength = None

        # =========================
        # ④ screening_conditions を完全削除
        # =========================
        # （何も書かない）

        # =========================
        # ⑤ python_condition の評価
        # =========================

        # --- 安全な eval ---
        def eval_python_condition(condition, ctx):
            try:
                return bool(eval(condition, {"__builtins__": None}, ctx))
            except:
                return False

        # --- 評価用コンテキスト ---
        context = {
            "ema20_vs_ema50": (ema20 - ema50) if ema20 and ema50 else None,
            "ema50_vs_ema200": (ema50 - ema200) if ema50 and ema200 else None,
            "price_vs_ema20_pct": (close_price / ema20 - 1) * 100 if ema20 else None,
            "drop_from_high_pct": drop_rate,
            "rebound_from_low_pct": reversal_rate,
            "vol_vs_ma20": volume_ratio,
            "atr_ratio": (atr / close_price * 100) if close_price else None,
        }

        # --- python_condition によるフィルタリング ---
        if python_condition and not eval_python_condition(python_condition, context):
            log(f"[FILTER] {symbol}: python_condition NG")
            return None

        # =========================
        # ⑥ スコア（日足のみ）
        # =========================
        short_score = (
            (reversal_strength or 0) * 0.4 +
            (volume_ratio or 0) * 0.2 +
            (slope_ema20 or 0) * 0.2 +
            (drop_rate or 0) * 0.1 -
            (atr or 0) * 0.1
        )

        mid_score = short_score  # 中期スコアは日足に統一

        # =========================
        # ⑦ GPT スコア（日足のみ）
        # =========================
        gpt = gpt_score(
            symbol, company_name, close_price, market_cap,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50, slope_ema20,
            atr, volume, vol_ma20, volume_ratio
        )

        # =========================
        # ⑧ 結果まとめ
        # =========================
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
            "slope_ema20": slope_ema20,
            "volume_ratio": volume_ratio,

            "ema50_mid": ema50,
            "slope_ema50_mid": slope_ema20,
            "drop_rate_mid": drop_rate,
            "reversal_rate_mid": reversal_rate,
            "reversal_strength_mid": reversal_strength,

            "short_score": short_score,
            "mid_score": mid_score,

            "gpt_score": gpt.get("score"),
            "gpt_judgement": gpt.get("judgement"),
            "gpt_comment": gpt.get("comment"),
            "passed_python_condition": True
        }

    except Exception as e:
        log(f"[ERROR] {symbol}: {e}")
        return None

# =========================
# メイン関数（screening ひとつだけ）
# =========================

@app.function_name(name="screening")
@app.route(route="screening", methods=["POST"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening start")

    logs = []

    try:
        # ① Blob 接続
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        # ② 結果保存先コンテナ
        result_container = os.getenv("RESULT_CONTAINER")

        # ③ ログ初期化
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

        # ④ log() 定義
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

        # ⑤ CSV 読み込み（Blob or Body）
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

        # ⑥ CSV → DataFrame
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

        # ⑦ デフォルト条件式
        default_python_condition = (
            "drop_from_high_pct < -15 and "
            "rebound_from_low_pct > 10 and "
            "vol_vs_ma20 > 1.5 and "
            "ema20 > ema50 and "
            "ema50 > ema200"
        )

        # ⑧ UI からの上書き
        ui_condition = req.headers.get("X-Python-Condition")
        python_condition = ui_condition if ui_condition else default_python_condition

        # ⑨ スクリーニング実行
        results = []

        for code in df_csv["コード"]:
            time.sleep(0.3)

            symbol = f"{code}.T"
            company_name = name_dict.get(code, "不明")
            market = market_dict.get(code, "不明")

            result = process_symbol(symbol, company_name, market, log, python_condition)

            if result is not None:
                results.append(result)

        # ⑩ JSON 保存（Blob）
        output_blob_name = f"{today}/{json_filename}"
        output_blob = blob_service.get_blob_client(result_container, output_blob_name)

        json_text = json.dumps(results, ensure_ascii=False, indent=2)
        output_blob.upload_blob(json_text, overwrite=True)

        log(f"[BLOB] JSON saved to {output_blob_name}")

        # =========================
        # ⑪ Azure AI Search へ登録
        # =========================
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

        # =========================
        # ⑫ レスポンス返却
        # =========================
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
# AI 買い候補ランキング生成
# =========================
@app.function_name(name="ranking")
@app.route(route="ranking", methods=["POST"], auth_level="anonymous")
def ranking(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        results = body.get("results", [])

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # --- SCORE 計算（短期 & 中期） ---
        for r in results:
            # 短期スコア（short_score）
            r["short_score"] = (
                (r.get("reversal_strength") or 0) * 0.4 +
                (r.get("volume_ratio") or 0) * 0.2 +
                (r.get("slope_ema20") or 0) * 0.2 +
                (r.get("drop_rate") or 0) * 0.1 -
                (r.get("ATR") or 0) * 0.1
            )

            # 中期スコア（mid_score）
            r["mid_score"] = (
                (r.get("reversal_strength_mid") or 0) * 0.5 +
                (r.get("slope_ema50_mid") or 0) * 0.3 +
                (r.get("drop_rate_mid") or 0) * 0.2
            )

        # --- GPT に渡す比較しやすい形式 ---
        items_text = ""
        for r in results:
            items_text += f"{r.get('symbol')} ({r.get('company_name')}): "
            items_text += f"short_score={r.get('short_score'):.2f}, "
            items_text += f"mid_score={r.get('mid_score'):.2f}, "
            items_text += f"reversal_strength={r.get('reversal_strength')}, "
            items_text += f"reversal_strength_mid={r.get('reversal_strength_mid')}, "
            items_text += f"slope_ema20={r.get('slope_ema20')}, "
            items_text += f"slope_ema50_mid={r.get('slope_ema50_mid')}, "
            items_text += f"drop_rate={r.get('drop_rate')}, "
            items_text += f"drop_rate_mid={r.get('drop_rate_mid')}, "
            items_text += f"volume_ratio={r.get('volume_ratio')}, "
            items_text += f"ATR={r.get('ATR')}\n"

        # --- GPT プロンプト（短期 × 中期 SCORE 最適化版） ---
        prompt = f"""
あなたは短期トレードの専門家です。

以下の銘柄データを比較し、
「買い候補トップ3」を選び、JSON 形式で出力してください。

【銘柄データ】
{items_text}

【評価基準】
1. 短期SCORE（short_score）
2. 中期SCORE（mid_score）
3. 反転強度（短期・中期）
4. 出来高急増率（volume_ratio）
5. ATR（リスクの低さ）
6. EMA20/EMA50 と EMA50_mid の整合性

【短期 × 中期の評価ルール】
・短期と中期の傾きが一致している銘柄は強く評価する
・短期が強く中期が弱い場合は「短期先行型」
・中期が強く短期が弱い場合は「中期主導型」
・反転強度の差が大きい場合は理由に含める
・EMA20/EMA50 と EMA50_mid の整合性を評価する

【観点割り当てルール】
・3銘柄の理由は必ず異なる観点を使うこと
　- 勢い（反転強度・EMA傾き）
　- 安定性（ATR・出来高）
　- 割安性（下落率）
　- トレンド（EMA20/EMA50）
・同じ観点を複数銘柄で使ってはならない

【理由の書き方】
・短期と中期の勢いの違いを必ず1文含める
・どちらが主導しているかを明確にする
・時間軸の整合性（同方向 or 乖離）を説明する
・理由は200文字以内
・リスクと注意点は100文字以内

【出力フォーマット（JSON のみ）】
{{
  "ranking": [
    {{
      "rank": 1,
      "symbol": "XXXX.T",
      "company": "銘柄名",
      "reason": "200文字以内（観点1つ）",
      "risk": "100文字以内",
      "note": "100文字以内"
    }},
    {{
      "rank": 2
    }},
    {{
      "rank": 3
    }}
  ]
}}
前後に説明文は書かず、JSON のみを返してください。
"""

        # --- GPT 呼び出し ---
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        ranking_json = json.loads(raw[json_start:json_end])

        return func.HttpResponse(
            json.dumps(ranking_json, ensure_ascii=False, indent=2),
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("ranking error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

# =========================
# 銘柄説明 API（RAG 本体）
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

        # --- Azure Search 接続 ---
        search_endpoint = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = "screening-results"

        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )

        # --- 最新データを検索（date の降順で 1 件） ---
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

        # --- GPT に説明させる ---
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
# 異常値アラート API（最終完成版）
# =========================
@app.function_name(name="alert_abnormal")
@app.route(route="alert_abnormal", methods=["GET"], auth_level="anonymous")
def alert_abnormal(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # --- Azure Search 接続 ---
        search_endpoint = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = "screening-results"

        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )

        # --- 全件取得（最大 1000 件） ---
        results = search_client.search(
            search_text="*",
            top=1000
        )

        docs = list(results)
        abnormal_list = []

        # --- 異常値条件 ---
        for doc in docs:
            r = json.loads(doc["json_text"])

            atr = r.get("ATR")
            volume_ratio = r.get("volume_ratio")
            drop_rate = r.get("drop_rate")
            reversal_strength = r.get("reversal_strength")

            is_abnormal = (
                (atr is not None and atr > 1.5) or
                (volume_ratio is not None and volume_ratio >= 2.0) or
                (drop_rate is not None and drop_rate <= -40) or
                (reversal_strength is not None and reversal_strength >= 2.0)
            )

            if is_abnormal:
                abnormal_list.append(r)

        # --- 429 対策：異常値が多い場合は 30 件に絞る ---
        if len(abnormal_list) > 30:
            abnormal_list = abnormal_list[:30]

        # --- AI に説明させる ---
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        prompt = f"""
あなたは短期トレードの専門家です。
以下の銘柄は「異常値」を検知した銘柄です。

異常値の種類：
- ATR が急上昇（ボラティリティ増加）
- volume_ratio ≥ 2.0（出来高急増）
- drop_rate ≤ -40%（急落）
- reversal_strength ≥ 2.0（反転強度が高い）

以下の JSON データを読み取り、
各銘柄について「なぜ異常なのか」「何に注意すべきか」を簡潔に説明してください。

{json.dumps(abnormal_list, ensure_ascii=False)}

出力形式：
[
  {{
    "symbol": "XXXX",
    "company": "企業名",
    "reason": "異常値の理由（100〜200文字）",
    "risk": "注意すべきリスク（50〜100文字）"
  }}
]
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        explanation = res.choices[0].message.content.strip()

        return func.HttpResponse(
            explanation,
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("alert_abnormal error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
