import azure.functions as func
import logging
import json
import yfinance as yf

app = func.FunctionApp()

@app.function_name(name="screening")
@app.route(route="screening", methods=["GET"], auth_level="anonymous")
def screening(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("screening minimal version start")

    symbol = req.params.get("symbol")
    if not symbol:
        return func.HttpResponse(
            json.dumps({"error": "symbol が指定されていません"}),
            mimetype="application/json",
            status_code=400
        )

    try:
        df = yf.download(symbol, period="5d", interval="1h")
        if df is None or df.empty:
            return func.HttpResponse(
                json.dumps({"error": f"{symbol} の株価データが取得できませんでした"}),
                mimetype="application/json",
                status_code=500
            )

        latest = df.iloc[-1]

        # --- Azure Functions で安全に float に変換 ---
        def safe_float(x):
            if hasattr(x, "iloc"):
                return float(x.iloc[0])
            return float(x)

        result = {
            "symbol": symbol,
            "close": safe_float(latest["Close"]),
            "high": safe_float(latest["High"]),
            "low": safe_float(latest["Low"]),
            "volume": int(safe_float(latest["Volume"]))
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
