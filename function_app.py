import azure.functions as func
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


# ---------------------------------------------------------
# 1. run_screening（スクリーニング実行）
# ---------------------------------------------------------
@app.route(route="run_screening", methods=["POST"])
def run_screening(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # TODO: shared/screening_logic.py を呼び出す
        result = {"status": "ok", "message": "screening executed"}
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


# ---------------------------------------------------------
# 2. run_monitoring（監視実行）
# ---------------------------------------------------------
@app.route(route="run_monitoring", methods=["POST"])
def run_monitoring(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # TODO: shared/monitoring_logic.py を呼び出す
        result = {"status": "ok", "message": "monitoring executed"}
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


# ---------------------------------------------------------
# 3. get_screening_latest（最新スクリーニング結果を取得）
# ---------------------------------------------------------
@app.route(route="get_screening_latest", methods=["GET"])
def get_screening_latest(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # TODO: shared/storage_utils.py から最新 HTML/JSON を取得
        result = {"status": "ok", "data": "latest screening data"}
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


# ---------------------------------------------------------
# 4. get_symbol_history（銘柄の過去データを取得）
# ---------------------------------------------------------
@app.route(route="get_symbol_history", methods=["GET"])
def get_symbol_history(req: func.HttpRequest) -> func.HttpResponse:
    try:
        symbol = req.params.get("symbol")
        if not symbol:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "symbol is required"}),
                status_code=400,
                mimetype="application/json"
            )

        # TODO: shared/yfinance_utils.py を呼び出す
        result = {"status": "ok", "symbol": symbol, "data": "history data"}
        return func.HttpResponse(json.dumps(result), mimetype="application/json")

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
