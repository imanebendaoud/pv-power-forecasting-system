"""
history.py — Gestionnaire d'historique SQLite
Sauvegarde automatique à chaque prévision :
  timestamp, pv_kw (multi-step), météo complète

Améliorations vs CSV :
  - SQLite : vraie base de données, requêtes SQL, plus rapide sur 1000+ lignes
  - HISTORY_DATA_DIR configurable (volume Docker)
  - save_record() accepte un timestamp optionnel (backfill)
  - last_timestamp() pour détecter les trous au démarrage
  - export_csv() génère un CSV à la volée depuis SQLite (compatibilité conservée)
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Permet de pointer vers un volume Docker monté (ex: /app/data)
# Sinon, fallback sur le dossier du code (dev local)
DATA_DIR = os.environ.get("HISTORY_DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE  = os.path.join(DATA_DIR, "historique.db")
CSV_FILE = os.path.join(DATA_DIR, "historique_export.csv")

# =====================================================
# INIT — crée la base et la table si absentes
# =====================================================
def init_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS previsions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp          TEXT    NOT NULL,
            pv_forecast_kw     REAL    NOT NULL,
            temperature        REAL,
            humidity           REAL,
            wind_speed         REAL,
            global_radiation   REAL,
            solar_elevation    REAL,
            module_temperature REAL,
            clearness_index    REAL,
            is_day             INTEGER,
            sky_label          TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON previsions(timestamp)
    """)
    conn.commit()
    conn.close()

# =====================================================
# SAUVEGARDER UNE LIGNE
# =====================================================
def save_record(pv_kw: float, weather: dict, sky_label: str, timestamp=None):
    """
    pv_kw     : prévision multi-step (point[0]) en kW
    weather   : dict retourné par /api/weather
    sky_label : label état du ciel (ex: "Très ensoleillé")
    timestamp : optionnel, pd.Timestamp ou datetime.
                Permet le backfill de créneaux passés.
                Si None, utilise l'heure actuelle.
    """
    init_history()

    ts     = timestamp if timestamp is not None else datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO previsions (
            timestamp, pv_forecast_kw,
            temperature, humidity, wind_speed,
            global_radiation, solar_elevation,
            module_temperature, clearness_index,
            is_day, sky_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts_str,
        round(float(pv_kw), 3),
        weather.get("temperature"),
        weather.get("humidity"),
        weather.get("wind_speed"),
        weather.get("global_radiation"),
        weather.get("solar_elevation"),
        weather.get("module_temperature"),
        weather.get("clearness_index"),
        weather.get("is_day"),
        sky_label,
    ))
    conn.commit()
    conn.close()

    print(f"[history] Enregistré : {ts_str} → {pv_kw:.2f} kW")

# =====================================================
# DERNIER TIMESTAMP ENREGISTRÉ
# Utilisé au démarrage pour détecter un trou et backfill
# =====================================================
def last_timestamp():
    """Retourne le dernier timestamp enregistré (pd.Timestamp) ou None."""
    init_history()

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM previsions")
        row = cursor.fetchone()
        conn.close()

        if row[0] is None:
            return None
        return pd.Timestamp(row[0])

    except Exception as e:
        print(f"[history] Erreur last_timestamp : {e}")
        return None

# =====================================================
# LIRE L'HISTORIQUE
# =====================================================
def load_history(days: int = 7) -> list:
    """
    Retourne les N derniers jours d'historique
    sous forme de liste de dicts.
    """
    init_history()

    try:
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")

        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT
                timestamp, pv_forecast_kw,
                temperature, humidity, wind_speed,
                global_radiation, solar_elevation,
                module_temperature, clearness_index,
                is_day, sky_label
            FROM previsions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff,))
        conn.close()

        if df.empty:
            return []

        return df.fillna("").to_dict(orient="records")

    except Exception as e:
        print(f"[history] Erreur lecture : {e}")
        return []

# =====================================================
# STATISTIQUES JOURNALIÈRES
# =====================================================
def daily_stats(days: int = 7) -> list:
    """
    Retourne les statistiques par jour :
    date, nb_records, pv_max, pv_mean, pv_total_kwh
    """
    init_history()

    try:
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")

        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT
                DATE(timestamp)          AS date,
                COUNT(*)                 AS nb_records,
                MAX(pv_forecast_kw)      AS pv_max,
                AVG(pv_forecast_kw)      AS pv_mean,
                SUM(pv_forecast_kw)*0.5  AS pv_total_kwh
            FROM previsions
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, conn, params=(cutoff,))
        conn.close()

        if df.empty:
            return []

        df["pv_max"]       = df["pv_max"].round(2)
        df["pv_mean"]      = df["pv_mean"].round(2)
        df["pv_total_kwh"] = df["pv_total_kwh"].round(2)

        return df.to_dict(orient="records")

    except Exception as e:
        print(f"[history] Erreur stats : {e}")
        return []

# =====================================================
# EXPORT CSV — génère un CSV depuis SQLite
# Compatibilité conservée avec /api/history/export
# =====================================================
def export_csv(days: int = 30) -> str:
    """
    Génère un fichier CSV depuis SQLite et retourne son chemin.
    """
    init_history()

    try:
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")

        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT
                timestamp, pv_forecast_kw,
                temperature, humidity, wind_speed,
                global_radiation, solar_elevation,
                module_temperature, clearness_index,
                is_day, sky_label
            FROM previsions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff,))
        conn.close()

        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        print(f"[history] Export CSV : {CSV_FILE} ({len(df)} lignes)")
        return CSV_FILE

    except Exception as e:
        print(f"[history] Erreur export CSV : {e}")
        return CSV_FILE