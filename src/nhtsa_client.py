import requests

NHTSA_BASE = "https://api.nhtsa.gov"


def execute_query(query: dict) -> dict:
    if "error" in query:
        return query

    endpoint = query.get("endpoint")
    make = query.get("make", "").upper()
    model = query.get("model", "").upper()
    year = query.get("year", "")

    try:
        if endpoint == "recalls":
            resp = requests.get(
                f"{NHTSA_BASE}/recalls/recallsByVehicle",
                params={"make": make, "model": model, "modelYear": year},
                timeout=10,
            )
        elif endpoint == "complaints":
            resp = requests.get(
                f"{NHTSA_BASE}/complaints/complaintsByVehicle",
                params={"make": make, "model": model, "modelYear": year},
                timeout=10,
            )
        elif endpoint == "safetyRatings":
            resp = requests.get(
                f"{NHTSA_BASE}/SafetyRatings/modelyear/{year}/make/{make}/model/{model}",
                timeout=10,
            )
        else:
            return {"error": "unknown_endpoint", "message": f"Unknown endpoint: {endpoint}"}

        resp.raise_for_status()
        return resp.json()

    except requests.Timeout:
        return {"error": "timeout", "message": "NHTSA API request timed out."}
    except requests.HTTPError as e:
        return {"error": "http_error", "message": str(e)}
    except requests.RequestException as e:
        return {"error": "api_error", "message": str(e)}
