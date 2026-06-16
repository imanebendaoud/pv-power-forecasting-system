import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from flask import Flask, jsonify, render_template, send_file, request
import pandas as pd
import traceback
from flasgger import Swagger
from apscheduler.schedulers.background import BackgroundScheduler

# Source unique : multi-step
from inference.forecast import forecast_12h, get_weather, fix_timeseries, preprocess
from history import save_record, load_history, daily_stats, export_csv

app = Flask(__name__)
swagger = Swagger(app)

# =====================================================
# HELPER
# =====================================================
def sky_state(radiation, is_day):
    if not is_day:
        return {"icon": "🌙", "label": "Nuit",            "level": "night"}
    if radiation > 700:
        return {"icon": "☀️",  "label": "Très ensoleillé", "level": "high"}
    if radiation > 400:
        return {"icon": "🌤",  "label": "Part. ensoleillé","level": "medium"}
    if radiation > 150:
        return {"icon": "⛅",  "label": "Nuageux",         "level": "low"}
    return     {"icon": "☁️",  "label": "Très nuageux",    "level": "vlow"}

def get_meteo_now():
    now = pd.Timestamp.now(tz="Africa/Casablanca")
    df  = get_weather()
    df  = fix_timeseries(df)
    df  = preprocess(df)
    idx = df.index.get_indexer([now], method="nearest")[0]
    row = df.iloc[idx]
    rad    = round(float(row["global_radiation"]), 1)
    is_day = int(row["is_day"])
    return df, idx, row, rad, is_day, sky_state(rad, is_day)

# =====================================================
# JOB PLANIFIÉ — sauvegarde automatique toutes les 30min
# Indépendant des pages ouvertes dans le navigateur
# =====================================================
def scheduled_forecast_save():
    try:
        points = forecast_12h()
        if not points:
            print("[scheduler] Aucune prévision disponible")
            return

        pv = points[0]["pv_kw"]

        _, _, row, rad, is_day, sky = get_meteo_now()
        weather_data = {
            "temperature":        round(float(row["ambient_temperature"]), 1),
            "humidity":           round(float(row["humidity"]), 1),
            "wind_speed":         round(float(row["wind_speed"]), 1),
            "global_radiation":   rad,
            "solar_elevation":    round(float(row["solar_elevation"]), 1),
            "is_day":             is_day,
            "module_temperature": round(float(row["module_temperature"]), 1),
            "clearness_index":    round(float(row["clearness_index"]), 3),
        }

        save_record(float(pv), weather_data, sky["label"])
        print(f"[scheduler] Enregistré automatiquement : {pv:.2f} kW")

    except Exception as e:
        print(f"[scheduler] Erreur : {e}")
        traceback.print_exc()

scheduler = BackgroundScheduler(timezone="Africa/Casablanca")
scheduler.add_job(
    scheduled_forecast_save,
    trigger="interval",
    minutes=30,
    id="forecast_autosave",
    next_run_time=pd.Timestamp.now(tz="Africa/Casablanca")  # exécute immédiatement au démarrage
)
scheduler.start()

# =====================================================
# PAGES
# =====================================================
@app.route("/")
def index():
    """Page principale dashboard
    ---
    tags: [Interface]
    responses:
      200:
        description: Dashboard HTML
    """
    return render_template("index.html")

@app.route("/historique")
def historique():
    """Page historique
    ---
    tags: [Interface]
    responses:
      200:
        description: Historique HTML
    """
    return render_template("historique.html")

# =====================================================
# API — FORECAST NOW
# Multi-step point[0] → même source que la courbe
# Sauvegarde redondante désactivée ici car le scheduler
# s'en occupe déjà — on lit juste la valeur
# =====================================================
@app.route("/api/forecast/now")
def api_forecast_now():
    """
    Prévision PV next 30min — VMD-LSTM Multi-Step
    ---
    tags: [Forecast]
    responses:
      200:
        description: PV next 30min
        schema:
          type: object
          properties:
            status:    {type: string, example: ok}
            timestamp: {type: string, example: "2026-06-13 14:30"}
            pv_kw:     {type: number, example: 32.451}
    """
    try:
        points = forecast_12h()
        if not points:
            raise ValueError("Aucune prévision")

        pv = points[0]["pv_kw"]
        ts = (pd.Timestamp.now(tz="Africa/Casablanca")
              + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")

        return jsonify({
            "status":    "ok",
            "timestamp": ts,
            "pv_kw":     round(float(pv), 3)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — COURBE 12H
# =====================================================
@app.route("/api/forecast/12h")
def api_forecast_12h():
    """
    Courbe 12h — VMD-LSTM Multi-Step
    ---
    tags: [Forecast]
    responses:
      200:
        description: Série PV 12h
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            points:
              type: array
              items:
                type: object
                properties:
                  time:  {type: string, example: "14:30"}
                  pv_kw: {type: number, example: 32.451}
    """
    try:
        points = forecast_12h()
        return jsonify({"status": "ok", "points": points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — MÉTÉO
# =====================================================
@app.route("/api/weather")
def api_weather():
    """Météo actuelle + état du ciel
    ---
    tags: [Weather]
    responses:
      200:
        description: Météo temps réel
        schema:
          type: object
          properties:
            status:             {type: string, example: ok}
            timestamp:          {type: string, example: "2026-06-13 14:30"}
            temperature:        {type: number, example: 28.4}
            humidity:           {type: number, example: 32.0}
            wind_speed:         {type: number, example: 14.2}
            global_radiation:   {type: number, example: 850.3}
            solar_elevation:    {type: number, example: 62.1}
            is_day:             {type: integer, example: 1}
            module_temperature: {type: number, example: 45.6}
            clearness_index:    {type: number, example: 0.93}
            clouds:             {type: number, example: 8.5}
            sky:
              type: object
              properties:
                icon:  {type: string, example: "☀️"}
                label: {type: string, example: "Très ensoleillé"}
                level: {type: string, example: "high"}
    """
    try:
        df, idx, row, rad, is_day, sky = get_meteo_now()
        return jsonify({
            "status":             "ok",
            "timestamp":          df.index[idx].strftime("%Y-%m-%d %H:%M"),
            "temperature":        round(float(row["ambient_temperature"]), 1),
            "humidity":           round(float(row["humidity"]), 1),
            "wind_speed":         round(float(row["wind_speed"]), 1),
            "global_radiation":   rad,
            "solar_elevation":    round(float(row["solar_elevation"]), 1),
            "is_day":             is_day,
            "module_temperature": round(float(row["module_temperature"]), 1),
            "clearness_index":    round(float(row["clearness_index"]), 3),
            "clouds":             round(float(row["clouds"]), 1),
            "sky":                sky,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — MÉTÉO 12H
# =====================================================
@app.route("/api/weather/12h")
def api_weather_12h():
    """Prévision météo 12h
    ---
    tags: [Weather]
    responses:
      200:
        description: Série météo 12h
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            chart:
              type: array
              items:
                type: object
                properties:
                  time:        {type: string, example: "15:00"}
                  radiation:   {type: number, example: 780.2}
                  temperature: {type: number, example: 29.1}
    """
    try:
        now = pd.Timestamp.now(tz="Africa/Casablanca")
        df  = get_weather()
        df  = fix_timeseries(df)
        df  = preprocess(df)
        now_idx = df.index.get_indexer([now], method="nearest")[0]
        rows = []
        for i in range(1, 25):
            idx = now_idx + i
            if idx >= len(df): break
            row = df.iloc[idx]
            rows.append({
                "time":        df.index[idx].strftime("%H:%M"),
                "radiation":   round(float(row["global_radiation"]),    1),
                "temperature": round(float(row["ambient_temperature"]), 1),
            })
        return jsonify({"status": "ok", "chart": rows})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — HISTORIQUE
# =====================================================
@app.route("/api/history")
def api_history():
    """Historique des prévisions
    ---
    tags: [Historique]
    parameters:
      - name: days
        in: query
        type: integer
        default: 7
        description: Nombre de jours d'historique à récupérer
    responses:
      200:
        description: Liste des enregistrements
        schema:
          type: object
          properties:
            status:  {type: string, example: ok}
            count:   {type: integer, example: 48}
            records:
              type: array
              items:
                type: object
                properties:
                  timestamp:          {type: string, example: "2026-06-13 14:30"}
                  pv_forecast_kw:     {type: number, example: 32.451}
                  temperature:        {type: number, example: 28.4}
                  humidity:           {type: number, example: 32.0}
                  wind_speed:         {type: number, example: 14.2}
                  global_radiation:   {type: number, example: 850.3}
                  solar_elevation:    {type: number, example: 62.1}
                  module_temperature: {type: number, example: 45.6}
                  clearness_index:    {type: number, example: 0.93}
                  is_day:             {type: integer, example: 1}
                  sky_label:          {type: string, example: "Très ensoleillé"}
    """
    try:
        days    = int(request.args.get("days", 7))
        records = load_history(days=days)
        return jsonify({"status": "ok", "records": records, "count": len(records)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — STATS JOURNALIÈRES
# =====================================================
@app.route("/api/history/stats")
def api_history_stats():
    """Statistiques journalières
    ---
    tags: [Historique]
    parameters:
      - name: days
        in: query
        type: integer
        default: 7
        description: Nombre de jours d'historique à analyser
    responses:
      200:
        description: Stats par jour
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            stats:
              type: array
              items:
                type: object
                properties:
                  date:         {type: string, example: "2026-06-13"}
                  nb_records:   {type: integer, example: 48}
                  pv_max:       {type: number, example: 52.3}
                  pv_mean:      {type: number, example: 24.1}
                  pv_total_kwh: {type: number, example: 289.2}
    """
    try:
        days = int(request.args.get("days", 7))
        return jsonify({"status": "ok", "stats": daily_stats(days=days)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# API — EXPORT CSV
# =====================================================
@app.route("/api/history/export")
def api_history_export():
    """Export CSV de l'historique
    ---
    tags: [Historique]
    produces:
      - text/csv
    responses:
      200:
        description: Fichier CSV téléchargeable
    """
    try:
        return send_file(export_csv(), mimetype="text/csv",
                         as_attachment=True,
                         download_name="historique_pvforecast.csv")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)