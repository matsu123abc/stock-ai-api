import azure.functions as func
import logging
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# ============================================================
# 1. Screening API
# ============================================================
@app.function_name(name="screening")
@app.route(route="screening", methods=["GET"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening start")

    # ★ ここに screening のロジックを移植
    # 例：
    # tickers = ["7203.T", "6758.T"]
    # df = fetch_price_data(tickers)
    # result = run_screening(df)

    result = {"status": "ok", "message": "screening done"}
    return func.HttpResponse(json.dumps(result), mimetype="application/json")


# ============================================================
# 2. Monitoring API
# ============================================================
@app.function_name(name="monitoring")
@app.route(route="monitoring", methods=["GET"], auth_level="anonymous")
def monitoring(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("monitoring start")

    # ★ ここに monitoring のロジックを移植

    result = {"status": "ok", "message": "monitoring done"}
    return func.HttpResponse(json.dumps(result), mimetype="application/json")


# ============================================================
# 3. 指標計算（例：反転強度）
# ============================================================
def calc_reversal_strength(df: pd.DataFrame) -> float:
    # ★ ここに indicators.py のロジックを移植
    return 0.0


# ============================================================
# 4. 株価データ取得（共通関数）
# ============================================================
def fetch_price_data(tickers):
    data = yf.download(tickers, period="1y")
    return data


# ============================================================
# 5. Blob 保存（共通関数）
# ============================================================
def save_to_blob(content: str, filename: str):
    conn_str = "<後で環境変数に移動>"
    container = "results"

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container, blob=filename)
    blob_client.upload_blob(content, overwrite=True)
