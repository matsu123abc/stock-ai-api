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


