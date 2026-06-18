# PV Power Forecasting System

## Overview

This project presents an intelligent photovoltaic power forecasting system developed as part of a Master's Degree Final Project (PFE) in Data Analytics and Artificial Intelligence.

The system predicts the future power production of the Noor PV I photovoltaic plant using advanced deep learning techniques and real-time meteorological data.

The forecasting pipeline combines:

* Variational Mode Decomposition (VMD)
* Long Short-Term Memory Networks (LSTM)
* Feature Selection using Random Forest
* Real-time weather data from Open-Meteo API
* Physical feature generation using PVLib
* Interactive web dashboard built with Flask

---

## Project Objectives

The main objective is to provide accurate short-term photovoltaic power forecasts that can support:

* Energy management
* Grid operation
* Renewable energy integration
* Decision-making for photovoltaic plant operators

---

## Technologies Used

### Artificial Intelligence & Data Science

* Python
* TensorFlow / Keras
* Scikit-Learn
* NumPy
* Pandas

### Photovoltaic Modeling

* PVLib

### Web Development

* Flask
* HTML5
* CSS3
* JavaScript

### Deployment

* Docker
* Docker Compose

---

## Forecasting Pipeline

1. Weather data acquisition from Open-Meteo API.
2. Temporal synchronization and preprocessing.
3. Generation of physical solar variables using PVLib.
4. Generation of derived meteorological features.
5. Data normalization using saved scalers.
6. Sliding window creation.
7. Prediction using optimized VMD-LSTM models.
8. Aggregation of VMD mode predictions.
9. Physical post-processing and consistency checks.
10. Visualization through a web dashboard.

---

## Model Architecture

The forecasting model is based on:

* VMD decomposition of historical PV production signals.
* Selection of the most informative modes.
* Dedicated LSTM model for each selected mode.
* Reconstruction of the final photovoltaic power forecast through aggregation of all predicted modes.

---

## Features

* Real-time weather integration.
* Multi-step photovoltaic power forecasting.
* Interactive dashboard.
* Forecast visualization charts.
* Dockerized deployment.
* Responsive web interface.

---

## Docker Deployment

Build and run:

```bash
docker-compose up --build
```

Access the application at:

```text
http://localhost:5000
```

---

## Author

Imane Ben Daoud

Master's Degree in Data Analytics and Artificial Intelligence (ADIA)

Final Year Project (PFE)

2025–2026
