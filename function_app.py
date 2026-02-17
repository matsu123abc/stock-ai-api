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

def process_symbol(symbol, company_name, market, log):
    try:
        # =========================
        # ① 短期データ（90日・1時間足）
        # =========================
        log(f"[DOWNLOAD-START] {symbol}: downloading 90d/1h data")

        df = yf.download(symbol, period="90d", interval="1h")

        if df is None or df.empty:
            log(f"[DOWNLOAD-WARN] {symbol}: no data returned")
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

        # --- 短期インジケータ ---
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

        # --- 短期の反転強度 ---
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
        # ② 中期データ（180日・日足）
        # =========================
        log(f"[DOWNLOAD-START] {symbol}: downloading 180d/1d data")

        df_mid = yf.download(symbol, period="180d", interval="1d")

        if df_mid is None or df_mid.empty:
            log(f"[DOWNLOAD-WARN] {symbol}: no mid-term data returned")

            # 中期データが無い場合は None を入れておく
            ema50_mid = None
            slope_ema50_mid = None
            drop_rate_mid = None
            reversal_rate_mid = None
            reversal_strength_mid = None

        else:
            log(f"[DOWNLOAD-END] {symbol}: {len(df_mid)} rows downloaded")

            # --- 中期 EMA ---
            df_mid["EMA50_mid"] = ema(df_mid["Close"], 50)

            ema50_mid = safe_float(df_mid["EMA50_mid"].iloc[-1])
            ema50_mid_prev = safe_float(df_mid["EMA50_mid"].iloc[-5])
            slope_ema50_mid = safe_float(ema50_mid - ema50_mid_prev)

            # --- 中期の反転強度（半年） ---
            recent_mid = df_mid.tail(120)
            peak_mid = safe_float(recent_mid["High"].max())
            bottom_mid = safe_float(recent_mid["Low"].min())

            drop_rate_mid = safe_float((bottom_mid / peak_mid - 1) * 100) if peak_mid else None
            reversal_rate_mid = safe_float((ema50_mid / bottom_mid - 1) * 100) if bottom_mid else None

            if drop_rate_mid and drop_rate_mid != 0:
                reversal_strength_mid = safe_float(reversal_rate_mid / abs(drop_rate_mid))
            else:
                reversal_strength_mid = None

        # =========================
        # ③ screening_conditions（短期のみ）
        # =========================
        if not screening_conditions(
            market_cap, close_price,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50
        ):
            log(f"[FILTER] {symbol}: screening_conditions NG")
            return None

        # =========================
        # ④ スコア（短期）
        # =========================
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

        # =========================
        # ⑤ GPT スコア（短期）
        # =========================
        gpt = gpt_score(
            symbol, company_name, close_price, market_cap,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50, slope_ema20,
            atr, volume, vol_ma20, volume_ratio
        )

        # =========================
        # ⑥ 結果まとめ（短期＋中期）
        # =========================
        return {
            "symbol": symbol,
            "company_name": company_name,
            "market": market,
            "close": close_price,

            # --- 短期 ---
            "EMA20": ema20,
            "EMA50": ema50,
            "ATR": atr,
            "drop_rate": drop_rate,
            "reversal_rate": reversal_rate,
            "reversal_strength": reversal_strength,
            "market_cap": market_cap,
            "slope_ema20": slope_ema20,
            "volume_ratio": volume_ratio,

            # --- 中期（半年） ---
            "ema50_mid": ema50_mid,
            "slope_ema50_mid": slope_ema50_mid,
            "drop_rate_mid": drop_rate_mid,
            "reversal_rate_mid": reversal_rate_mid,
            "reversal_strength_mid": reversal_strength_mid,

            # --- スコア ---
            "score": score,
            "gpt_score": gpt.get("score"),
            "gpt_judgement": gpt.get("judgement"),
            "gpt_comment": gpt.get("comment")
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

            result = process_symbol(symbol, company_name, market, log)

            if result is not None:
                results.append(result)


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

        # --- GPT に渡す比較しやすい形式 ---
        items_text += (
            f"{r.get('symbol')} ({r.get('company_name')}): "
            f"score={r.get('score')}, "
            f"gpt_score={r.get('gpt_score')}, "
            f"drop_rate={r.get('drop_rate')}, "
            f"reversal_rate={r.get('reversal_rate')}, "
            f"reversal_strength={r.get('reversal_strength')}, "
            f"EMA20={r.get('EMA20')}, "
            f"EMA50={r.get('EMA50')}, "
            f"slope_ema20={r.get('slope_ema20')}, "
            f"ATR={r.get('ATR')}, "
            f"volume_ratio={r.get('volume_ratio')}, "
            f"ema50_mid={r.get('ema50_mid')}, "
            f"slope_ema50_mid={r.get('slope_ema50_mid')}, "
            f"drop_rate_mid={r.get('drop_rate_mid')}, "
            f"reversal_rate_mid={r.get('reversal_rate_mid')}, "
            f"reversal_strength_mid={r.get('reversal_strength_mid')}\n"
        )

        prompt = f"""
あなたは短期トレードの専門家です。

以下の銘柄データを比較し、
「買い候補トップ3」を選び、JSON 形式で出力してください。

【銘柄データ】
{items_text}

【評価基準】
1. 反転強度（最重要）
2. 出来高急増率（volume_ratio）
3. ATR（リスクの低さ）
4. EMA20 と EMA50 の位置関係
5. EMA20 の傾き（slope_ema20）
6. score と gpt_score の総合点

【観点割り当てルール（最重要）】
・3銘柄の理由は、必ず異なる観点を使うこと
・以下の観点から「その銘柄に最も適した1つだけ」を選んで理由を書くこと
　- 勢い（反転強度・EMA傾き）
　- 安定性（ATR・出来高）
　- 割安性（下落率・反転率）
　- トレンド（EMA20/EMA50）
・同じ観点を複数銘柄で使ってはならない
・短期指標に加えて、中期（半年）の勢いも評価すること：
  - ema50_mid, slope_ema50_mid
  - drop_rate_mid, reversal_strength_mid
・短期と中期の勢いが一致している銘柄は高く評価すること
・短期は強いが中期は弱い銘柄、中期は強いが短期は弱い銘柄など、
  時間軸のギャップも理由に含めること

【順位ごとの役割分担】
1位：最も強い攻めの理由（勢い・優位性）
2位：強みと弱みのバランス型理由
3位：リスクを踏まえた上で条件次第で狙える理由

【文章ルール】
・理由は200文字以内
・リスクと注意点は100文字以内
・同じ表現や文章構造を繰り返さないこと

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
      "rank": 2, ... }},
    {{
      "rank": 3, ... }}
  ]
}}
前後に説明文は書かず、JSON のみを返してください。
"""

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
