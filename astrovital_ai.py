# =============================================================================
# AstroVital AI — Real-Time Astronaut Vitals Anomaly Detection
# =============================================================================
# Author      : Abraham
# Course      : ITAI 2372 — Final Project
# Track       : Implementation Track
# File        : astrovital_ai.py
#
# Description :
#   This is the main module for AstroVital AI. It implements a complete
#   machine learning pipeline to detect anomalies in astronaut biometric
#   data in real time.
#
#   The pipeline includes:
#     1. Synthetic biometric data generation  (simulates wearable sensors)
#     2. Data preprocessing                   (cleaning, normalization, features)
#     3. Anomaly detection model              (Isolation Forest — unsupervised)
#     4. Anomaly classification               (Random Forest — supervised)
#     5. Tiered alert system                  (CRITICAL / MEDIUM / LOW / NORMAL)
#     6. Results visualization                (charts saved to /output/)
#
#   Run this file directly to execute the full demo pipeline:
#       python astrovital_ai.py
#
# Dependencies: numpy, pandas, scikit-learn, matplotlib
#   Install with: pip install -r requirements.txt
# =============================================================================

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Seed for reproducibility — same seed = same results every run
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Output directory for charts and results
OUTPUT_DIR = "output"

# Normal biometric ranges (resting adult astronaut)
NORMAL_RANGES = {
    "heart_rate":  (60, 100),    # BPM
    "spo2":        (95, 100),    # % blood oxygen saturation
    "gsr":         (1.0, 5.0),   # microsiemens — galvanic skin response (stress)
    "skin_temp":   (36.0, 37.5), # degrees Celsius
}

# Number of readings to simulate
N_BASELINE  = 200   # normal readings used to train the detector
N_TEST      = 100   # normal test readings
N_ANOMALIES = 30    # anomaly readings injected into test set

# Alert severity labels with emoji for clear console output
SEVERITY = {
    "critical": "CRITICAL",
    "medium":   "MEDIUM",
    "low":      "LOW",
    "normal":   "NORMAL",
}


# =============================================================================
# SECTION 2 — SETUP: OUTPUT DIRECTORY
# =============================================================================

def setup_output_dir():
    """
    Create the output/ directory if it does not already exist.
    All charts and result files are saved here.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[SETUP] Output directory ready: ./{OUTPUT_DIR}/")
    except OSError as e:
        # Not fatal — warn the user but continue running
        print(f"[WARNING] Could not create output directory: {e}")
        print("          Charts will not be saved.")


# =============================================================================
# SECTION 3 — SYNTHETIC DATA GENERATOR
# =============================================================================
# In a real deployment, data would come from physical wearable sensors.
# Here we generate realistic synthetic biometric streams so the full
# pipeline can be demonstrated without hardware.

class BiometricDataGenerator:
    """
    Simulates a continuous stream of astronaut biometric sensor readings.

    Supports:
      - Normal readings with realistic Gaussian noise around a personal baseline
      - Activity-adjusted readings (exercise, sleep, work)
      - Injected anomaly scenarios for testing detection accuracy
    """

    def __init__(self, astronaut_name="Abraham_ISS01"):
        """
        Initialize the generator with a personalized baseline.
        Each astronaut has slightly different normal values (simulates
        real pre-mission biometric calibration).

        Parameters:
            astronaut_name (str): Identifier label for this astronaut
        """
        self.name = astronaut_name

        # Randomize personal baseline within realistic healthy adult ranges
        self.baseline = {
            "heart_rate": random.uniform(63, 74),
            "spo2":       random.uniform(96.5, 98.5),
            "gsr":        random.uniform(1.5, 2.8),
            "skin_temp":  random.uniform(36.2, 36.9),
        }

        print(f"\n[DATA GEN] Astronaut: {self.name}")
        print(f"           Personal baseline (from pre-mission calibration):")
        print(f"             Heart Rate : {self.baseline['heart_rate']:.1f} BPM")
        print(f"             SpO2       : {self.baseline['spo2']:.1f} %")
        print(f"             GSR        : {self.baseline['gsr']:.2f} uS")
        print(f"             Skin Temp  : {self.baseline['skin_temp']:.2f} C")

    def _apply_activity(self, hr, spo2, gsr, activity):
        """
        Adjust biometric values based on current activity.
        For example, exercise raises HR and GSR naturally — not an anomaly.

        Parameters:
            hr, spo2, gsr (float): Current biometric values
            activity (str): One of 'resting', 'exercising', 'working', 'sleeping'

        Returns:
            Tuple (hr, spo2, gsr) adjusted for activity
        """
        if activity == "exercising":
            hr   += random.uniform(30, 60)
            gsr  += random.uniform(2.0, 4.0)
            spo2 -= random.uniform(0.0, 1.5)
        elif activity == "sleeping":
            hr   -= random.uniform(5, 15)
            gsr  -= random.uniform(0.3, 0.8)
        elif activity == "working":
            hr   += random.uniform(5, 12)
            gsr  += random.uniform(0.5, 1.5)
        # 'resting' needs no adjustment
        return hr, spo2, gsr

    def generate_normal(self, n=100, activity="resting"):
        """
        Generate n normal biometric readings with realistic noise.

        Parameters:
            n        (int): Number of readings to generate
            activity (str): Activity context for all readings

        Returns:
            pd.DataFrame with columns: timestamp, heart_rate, spo2,
                                        gsr, skin_temp, activity, label
        """
        records = []
        for i in range(n):
            # Add Gaussian noise around personal baseline
            hr   = self.baseline["heart_rate"] + np.random.normal(0, 2.5)
            spo2 = self.baseline["spo2"]        + np.random.normal(0, 0.4)
            gsr  = self.baseline["gsr"]         + np.random.normal(0, 0.3)
            temp = self.baseline["skin_temp"]   + np.random.normal(0, 0.15)

            # Apply activity-based adjustments
            hr, spo2, gsr = self._apply_activity(hr, spo2, gsr, activity)

            # Clamp to physically realistic bounds
            records.append({
                "timestamp":  i,
                "heart_rate": round(float(np.clip(hr,   30,  220)), 1),
                "spo2":       round(float(np.clip(spo2,  85, 100)), 1),
                "gsr":        round(float(np.clip(gsr,  0.1, 25)),  2),
                "skin_temp":  round(float(np.clip(temp,  35,  42)), 2),
                "activity":   activity,
                "label":      "normal",
            })
        return pd.DataFrame(records)

    def generate_anomaly(self, anomaly_type, n=10):
        """
        Generate biometric readings containing a specific anomaly event.

        Parameters:
            anomaly_type (str): One of 'hypoxia', 'cardiac', 'stress', 'fatigue'
            n            (int): Number of anomaly readings to generate

        Returns:
            pd.DataFrame with anomalous readings and correct labels

        Raises:
            ValueError: If an unknown anomaly_type is provided
        """
        # Validate the anomaly type before generating data
        valid_types = ["hypoxia", "cardiac", "stress", "fatigue"]
        if anomaly_type not in valid_types:
            raise ValueError(
                f"Unknown anomaly type: '{anomaly_type}'. "
                f"Valid options: {valid_types}"
            )

        records = []
        for i in range(n):
            # Start from personal baseline with minimal noise
            hr   = self.baseline["heart_rate"] + np.random.normal(0, 1.5)
            spo2 = self.baseline["spo2"]        + np.random.normal(0, 0.3)
            gsr  = self.baseline["gsr"]         + np.random.normal(0, 0.2)
            temp = self.baseline["skin_temp"]   + np.random.normal(0, 0.1)

            # Inject the specific anomaly pattern on top of baseline
            if anomaly_type == "hypoxia":
                # Blood oxygen drops dangerously; heart rate compensates by rising
                spo2  = random.uniform(83, 91)
                hr   += random.uniform(20, 40)

            elif anomaly_type == "cardiac":
                # Dangerous tachycardia (very fast heart rate) at rest
                hr    = random.uniform(140, 185)

            elif anomaly_type == "stress":
                # Acute psychological stress: GSR spike + elevated HR
                gsr  += random.uniform(6, 12)
                hr   += random.uniform(25, 45)

            elif anomaly_type == "fatigue":
                # Chronic fatigue: lower HR, elevated temperature, GSR slightly high
                hr   -= random.uniform(10, 20)
                temp += random.uniform(0.5, 1.2)
                gsr  += random.uniform(1.0, 3.0)

            records.append({
                "timestamp":  i,
                "heart_rate": round(float(np.clip(hr,   30,  220)), 1),
                "spo2":       round(float(np.clip(spo2,  70, 100)), 1),
                "gsr":        round(float(np.clip(gsr,  0.1, 25)),  2),
                "skin_temp":  round(float(np.clip(temp,  35,  42)), 2),
                "activity":   "resting",
                "label":      anomaly_type,
            })
        return pd.DataFrame(records)


# =============================================================================
# SECTION 4 — DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """
    Cleans and prepares raw biometric data for the ML models.

    Pipeline steps:
      1. Validate — detect physically impossible sensor values (faults)
      2. Impute   — replace invalid/missing values with column median
      3. Features — add derived features (HR/SpO2 ratio, GSR x HR product)
      4. Scale    — normalize all features to [0, 1] range (MinMaxScaler)
    """

    def __init__(self):
        self.scaler      = MinMaxScaler()
        self.fitted      = False   # True after fit_transform() has been called
        self.feature_cols = ["heart_rate", "spo2", "gsr", "skin_temp"]
        print("\n[PREPROCESSOR] Initialized.")

    def validate(self, df):
        """
        Check for physically impossible sensor values and replace with NaN.
        These typically indicate sensor hardware faults, not medical events.

        Parameters:
            df (pd.DataFrame): Raw biometric readings

        Returns:
            pd.DataFrame with invalid values set to NaN
        """
        df    = df.copy()
        issues = 0

        # SpO2 at or below 5% is a sensor fault (not survivable in reality)
        mask_spo2 = (df["spo2"] <= 5) | (df["spo2"] > 100)
        if mask_spo2.any():
            print(f"  [VALIDATE] {mask_spo2.sum()} SpO2 fault(s) detected → NaN")
            df.loc[mask_spo2, "spo2"] = np.nan
            issues += int(mask_spo2.sum())

        # Heart rate outside 20–250 BPM is physically impossible
        mask_hr = (df["heart_rate"] < 20) | (df["heart_rate"] > 250)
        if mask_hr.any():
            print(f"  [VALIDATE] {mask_hr.sum()} HR fault(s) detected → NaN")
            df.loc[mask_hr, "heart_rate"] = np.nan
            issues += int(mask_hr.sum())

        if issues == 0:
            print("  [VALIDATE] All sensor values within physical bounds.")

        return df

    def impute(self, df):
        """
        Replace NaN values with the column median.
        Median is preferred over mean because it is robust to extreme outliers.

        Parameters:
            df (pd.DataFrame): Data with possible NaN values

        Returns:
            pd.DataFrame with no missing values
        """
        df = df.copy()
        for col in self.feature_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                median_val = df[col].median()
                df[col]    = df[col].fillna(median_val)
                print(f"  [IMPUTE] '{col}': {n_missing} NaN(s) "
                      f"replaced with median = {median_val:.2f}")
        return df

    def add_features(self, df):
        """
        Create derived features that help distinguish anomaly types.

        New features:
          hr_spo2_ratio  : HR / SpO2 — elevated when hypoxia causes compensatory HR rise
          temp_deviation : Distance from 36.5C (normal core temp center)
          gsr_hr_product : GSR * HR — combined stress + exertion signal

        Parameters:
            df (pd.DataFrame): Cleaned readings

        Returns:
            pd.DataFrame with three additional feature columns
        """
        df = df.copy()
        # Avoid division by zero in SpO2 ratio
        df["hr_spo2_ratio"]  = df["heart_rate"] / df["spo2"].replace(0, np.nan).fillna(1)
        df["temp_deviation"] = (df["skin_temp"] - 36.5).abs()
        df["gsr_hr_product"] = df["gsr"] * df["heart_rate"]
        return df

    def fit_transform(self, df):
        """
        Full preprocessing on TRAINING data.
        Fits the MinMaxScaler on this data, then scales all feature columns to [0,1].

        Parameters:
            df (pd.DataFrame): Training data

        Returns:
            pd.DataFrame with all features scaled
        """
        df         = self.validate(df)
        df         = self.impute(df)
        df         = self.add_features(df)
        scale_cols = self.feature_cols + ["hr_spo2_ratio", "temp_deviation", "gsr_hr_product"]

        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        self.fitted    = True
        print(f"  [SCALE] Scaler fitted and applied to {len(scale_cols)} features.")
        return df

    def transform(self, df):
        """
        Preprocess NEW data using the already-fitted scaler.
        Must call fit_transform() first.

        Parameters:
            df (pd.DataFrame): New readings to preprocess

        Returns:
            pd.DataFrame with scaled features

        Raises:
            RuntimeError: If called before fit_transform()
        """
        if not self.fitted:
            raise RuntimeError(
                "[PREPROCESSOR] Scaler not fitted yet. "
                "Call fit_transform() on training data first."
            )
        df         = self.validate(df)
        df         = self.impute(df)
        df         = self.add_features(df)
        scale_cols = self.feature_cols + ["hr_spo2_ratio", "temp_deviation", "gsr_hr_product"]
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df


# =============================================================================
# SECTION 5 — ANOMALY DETECTOR (Isolation Forest)
# =============================================================================
# Isolation Forest is an unsupervised ML algorithm for anomaly detection.
# It works by randomly partitioning features with decision trees. Anomalies
# are isolated with fewer splits (shorter path = more anomalous).
# Key advantage: only needs NORMAL data for training — no labeled anomalies.

class AnomalyDetector:
    """
    Unsupervised anomaly detector using scikit-learn's Isolation Forest.

    Trained only on normal biometric data. At inference time, it assigns
    an anomaly score to each reading and flags statistical outliers.
    """

    def __init__(self, contamination=0.08):
        """
        Parameters:
            contamination (float): Expected fraction of anomalies in new data.
                                   0.08 = expect ~8% of readings to be anomalous.
        """
        self.model = IsolationForest(
            n_estimators=100,          # 100 trees in the forest
            contamination=contamination,
            random_state=RANDOM_SEED,
            max_samples="auto"         # auto = min(256, n_samples)
        )
        self.fitted       = False
        self.feature_cols = None
        print(f"\n[DETECTOR] Isolation Forest initialized "
              f"(n_estimators=100, contamination={contamination})")

    def train(self, df, feature_cols):
        """
        Fit the Isolation Forest on preprocessed normal baseline data.

        Parameters:
            df           (pd.DataFrame): Preprocessed normal training data
            feature_cols (list of str):  Which columns to use as features

        Raises:
            Exception: Re-raises any sklearn fitting error with a helpful message
        """
        try:
            X = df[feature_cols].values
            self.model.fit(X)
            self.feature_cols = feature_cols
            self.fitted       = True
            print(f"  [DETECTOR] Trained on {len(X)} normal baseline readings.")
        except Exception as e:
            print(f"  [ERROR] Detector training failed: {e}")
            raise

    def predict(self, df):
        """
        Score new readings. Anomalies receive is_anomaly = True.

        Isolation Forest output:
          +1 = inlier  (normal)
          -1 = outlier (anomaly)

        Parameters:
            df (pd.DataFrame): Preprocessed readings to score

        Returns:
            pd.DataFrame with two new columns:
              'is_anomaly'    (bool)   — True if flagged as anomalous
              'anomaly_score' (float)  — lower value = more anomalous

        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self.fitted:
            raise RuntimeError(
                "[DETECTOR] Model not trained. Call train() first."
            )
        try:
            X              = df[self.feature_cols].values
            flags          = self.model.predict(X)        # +1 or -1
            scores         = self.model.score_samples(X)  # negative = more anomalous

            result                  = df.copy()
            result["is_anomaly"]    = (flags == -1)
            result["anomaly_score"] = np.round(scores, 5)
            n_flagged = int((flags == -1).sum())
            print(f"  [DETECTOR] {n_flagged} anomalies flagged "
                  f"out of {len(df)} readings.")
            return result
        except Exception as e:
            print(f"  [ERROR] Detector prediction failed: {e}")
            raise


# =============================================================================
# SECTION 6 — ANOMALY CLASSIFIER (Random Forest)
# =============================================================================
# Once an anomaly is flagged by the detector, the Random Forest classifier
# determines the TYPE and SEVERITY. This is a supervised model trained on
# labeled examples of each medical event category.

class AnomalyClassifier:
    """
    Supervised classifier using scikit-learn's Random Forest.

    Predicts:
      - Anomaly type  : hypoxia / cardiac / stress / fatigue / normal
      - Severity      : critical / medium / low / normal
    """

    # Map from anomaly type to alert severity
    SEVERITY_MAP = {
        "hypoxia":  "critical",
        "cardiac":  "critical",
        "stress":   "medium",
        "fatigue":  "low",
        "normal":   "normal",
    }

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_SEED,
            class_weight="balanced"    # handles class imbalance automatically
        )
        self.fitted       = False
        self.feature_cols = None
        print(f"\n[CLASSIFIER] Random Forest initialized "
              f"(n_estimators=100, max_depth=8)")

    def train(self, df, feature_cols):
        """
        Fit the Random Forest on labeled training data.

        Parameters:
            df           (pd.DataFrame): Data with 'label' column
            feature_cols (list of str):  Feature column names

        Raises:
            ValueError:  If fewer than 2 unique classes are present
            Exception:   Re-raises other sklearn errors
        """
        try:
            X             = df[feature_cols].values
            y             = df["label"].values
            unique_labels = np.unique(y)

            if len(unique_labels) < 2:
                raise ValueError(
                    f"Classifier needs at least 2 classes. Found: {unique_labels}"
                )

            self.model.fit(X, y)
            self.feature_cols = feature_cols
            self.fitted       = True
            print(f"  [CLASSIFIER] Trained on {len(X)} samples.")
            print(f"               Classes: {list(unique_labels)}")

        except Exception as e:
            print(f"  [ERROR] Classifier training failed: {e}")
            raise

    def predict(self, df):
        """
        Predict anomaly type and severity for all readings.

        Parameters:
            df (pd.DataFrame): Preprocessed readings

        Returns:
            pd.DataFrame with added columns:
              'predicted_type'  (str)   — anomaly category
              'confidence'      (float) — model confidence 0–1
              'severity'        (str)   — critical / medium / low / normal
              'severity_label'  (str)   — display label with emoji

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.fitted:
            raise RuntimeError(
                "[CLASSIFIER] Model not trained. Call train() first."
            )
        try:
            X            = df[self.feature_cols].values
            pred_types   = self.model.predict(X)
            pred_proba   = self.model.predict_proba(X).max(axis=1)

            result                   = df.copy()
            result["predicted_type"] = pred_types
            result["confidence"]     = np.round(pred_proba, 3)
            result["severity"]       = pd.Series(pred_types).map(self.SEVERITY_MAP).values
            result["severity_label"] = pd.Series(result["severity"].tolist()).map(SEVERITY).values
            return result

        except Exception as e:
            print(f"  [ERROR] Classifier prediction failed: {e}")
            raise

    def evaluate(self, df, feature_cols):
        """
        Print a full classification report (precision, recall, F1) on test data.

        Parameters:
            df           (pd.DataFrame): Test data with a 'label' column
            feature_cols (list of str):  Feature columns
        """
        try:
            X    = df[feature_cols].values
            y    = df["label"].values
            pred = self.model.predict(X)
            print("\n[CLASSIFIER] Evaluation on test set:")
            print("-" * 52)
            print(classification_report(y, pred, zero_division=0))
        except Exception as e:
            print(f"  [WARNING] Evaluation skipped: {e}")


# =============================================================================
# SECTION 7 — ALERT SYSTEM
# =============================================================================

class AlertSystem:
    """
    Generates tiered health alerts based on detected anomaly severity.

    Severity tiers:
      CRITICAL — Immediate intervention required  (cardiac, hypoxia)
      MEDIUM   — Crew notification + monitoring   (acute stress)
      LOW      — Log and watch for escalation     (fatigue)
      NORMAL   — No action needed
    """

    ACTIONS = {
        "critical": ("STOP current activity. Alert crew immediately. "
                     "Administer O2 if hypoxia suspected. Contact mission control."),
        "medium":   ("Increase monitoring frequency. Notify crew. "
                     "Consider scheduling rest. Log event."),
        "low":      ("Log event. Watch for trend escalation. "
                     "Recommend rest at next opportunity."),
        "normal":   "No action required.",
    }

    def __init__(self):
        self.alert_log = []  # All alerts stored for session summary
        print("\n[ALERT SYSTEM] Initialized. Monitoring active.")

    def process(self, row):
        """
        Process one classified reading and generate an alert if non-normal.

        Parameters:
            row (pd.Series): One row from the classified results DataFrame

        Returns:
            dict: Alert record if anomalous, or None if normal
        """
        severity = row.get("severity", "normal")

        # Only create alerts for readings flagged as non-normal
        if severity == "normal":
            return None

        alert = {
            "time":       time.strftime("%H:%M:%S"),
            "astronaut":  row.get("astronaut", "Unknown"),
            "type":       row.get("predicted_type", "unknown"),
            "severity":   row.get("severity_label", SEVERITY["normal"]),
            "hr":         row.get("heart_rate", "N/A"),
            "spo2":       row.get("spo2", "N/A"),
            "gsr":        row.get("gsr", "N/A"),
            "temp":       row.get("skin_temp", "N/A"),
            "confidence": row.get("confidence", 0.0),
            "action":     self.ACTIONS.get(severity, "Monitor and log."),
        }
        self.alert_log.append(alert)
        return alert

    def print_alert(self, alert):
        """Print a formatted alert box to the console."""
        if alert is None:
            return
        print(f"\n  {'='*56}")
        print(f"  ALERT — {alert['severity']}")
        print(f"  {'='*56}")
        print(f"  Time        : {alert['time']}")
        print(f"  Astronaut   : {alert['astronaut']}")
        print(f"  Event Type  : {alert['type'].upper()}")
        print(f"  Confidence  : {alert['confidence']*100:.0f}%")
        print(f"  Vitals      : HR={alert['hr']} BPM | SpO2={alert['spo2']}% "
              f"| GSR={alert['gsr']} uS | Temp={alert['temp']}C")
        print(f"  Action      : {alert['action']}")
        print(f"  {'='*56}")

    def summary(self):
        """Print a count summary of all alerts generated this session."""
        print(f"\n{'='*56}")
        print("  ALERT SESSION SUMMARY")
        print(f"{'='*56}")

        if not self.alert_log:
            print("  No alerts generated. All readings normal.")
        else:
            # Count by severity keyword in the label string
            counts = {"critical": 0, "medium": 0, "low": 0}
            for a in self.alert_log:
                for level in counts:
                    if level in a["severity"].lower():
                        counts[level] += 1

            print(f"  Total Alerts  : {len(self.alert_log)}")
            print(f"  CRITICAL      : {counts['critical']}")
            print(f"  MEDIUM        : {counts['medium']}")
            print(f"  LOW           : {counts['low']}")
        print(f"{'='*56}\n")


# =============================================================================
# SECTION 8 — VISUALIZATION
# =============================================================================

def plot_vitals_with_anomalies(results_df, astronaut_name="Astronaut"):
    """
    Plot all four biometric signals as time series with anomaly regions
    shaded by severity. Saves to output/vitals_chart.png

    Parameters:
        results_df     (pd.DataFrame): Full results with 'is_anomaly', 'severity' cols
        astronaut_name (str):          Name label for chart title
    """
    try:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"AstroVital AI — Biometric Monitor: {astronaut_name}",
            fontsize=14, fontweight="bold", color="#1F4E79"
        )

        signals = [
            ("heart_rate", "Heart Rate (BPM)", "#E74C3C",  (30,  220)),
            ("spo2",       "SpO2 (%)",          "#2E75B6",  (80,  101)),
            ("gsr",        "GSR (uS)",           "#27AE60",  (0,   20)),
            ("skin_temp",  "Skin Temp (C)",      "#E67E22",  (35,   42)),
        ]

        x = range(len(results_df))

        for ax, (col, label, color, ylim) in zip(axes, signals):
            ax.plot(x, results_df[col].values, color=color,
                    linewidth=1.2, alpha=0.85, label=label)

            # Shade anomaly regions by severity
            for i, row in results_df.iterrows():
                if row.get("is_anomaly", False):
                    sev   = str(row.get("severity", "low")).lower()
                    shade = ("red"    if "critical" in sev else
                             "orange" if "medium"   in sev else "yellow")
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.25, color=shade)

            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Reading Index (time)", fontsize=9)

        # Legend patches for anomaly shading
        legend_patches = [
            mpatches.Patch(color="red",    alpha=0.4, label="Critical Anomaly"),
            mpatches.Patch(color="orange", alpha=0.4, label="Medium Anomaly"),
            mpatches.Patch(color="yellow", alpha=0.4, label="Low Anomaly"),
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=3, fontsize=9, framealpha=0.8)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = os.path.join(OUTPUT_DIR, "vitals_chart.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [CHART] vitals_chart.png saved to ./{OUTPUT_DIR}/")

    except Exception as e:
        print(f"  [WARNING] Could not save vitals chart: {e}")


def plot_anomaly_distribution(results_df):
    """
    Bar chart showing how many readings were classified as each type.
    Saves to output/anomaly_distribution.png

    Parameters:
        results_df (pd.DataFrame): Results with 'predicted_type' column
    """
    try:
        type_counts = results_df["predicted_type"].value_counts()

        color_map = {
            "normal":  "#2E75B6",
            "hypoxia": "#C0392B",
            "cardiac": "#E74C3C",
            "stress":  "#F39C12",
            "fatigue": "#27AE60",
        }
        bar_colors = [color_map.get(t, "#7F8C8D") for t in type_counts.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(type_counts.index, type_counts.values,
                      color=bar_colors, edgecolor="white", linewidth=0.8)

        ax.set_title("AstroVital AI — Anomaly Type Distribution",
                     fontsize=13, fontweight="bold", color="#1F4E79")
        ax.set_xlabel("Detected Type",         fontsize=10)
        ax.set_ylabel("Number of Readings",    fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Label each bar with its count above the bar
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "anomaly_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [CHART] anomaly_distribution.png saved to ./{OUTPUT_DIR}/")

    except Exception as e:
        print(f"  [WARNING] Could not save distribution chart: {e}")


# =============================================================================
# SECTION 9 — MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """
    Execute the complete AstroVital AI pipeline end-to-end.

    Pipeline steps:
      1. Generate synthetic biometric training data
      2. Preprocess (validate, impute, scale, feature engineering)
      3. Train Isolation Forest anomaly detector on normal data
      4. Train Random Forest classifier on labeled anomaly data
      5. Generate mixed test set (normal + injected anomalies)
      6. Run detection + classification on test set
      7. Evaluate classifier performance (precision, recall, F1)
      8. Fire and display alerts for flagged readings
      9. Save charts to output/
    """

    print("\n" + "="*60)
    print("   AstroVital AI — Mission Health Monitor")
    print("   Real-Time Astronaut Vitals Anomaly Detection")
    print("   Author: Abraham | ITAI 2372")
    print("="*60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    setup_output_dir()

    # ── Step 1: Generate training data ────────────────────────────────────────
    print("\n[STEP 1] Generating synthetic biometric data...")
    gen = BiometricDataGenerator(astronaut_name="Abraham_ISS01")

    # Baseline: only normal readings — used to train the unsupervised detector
    baseline_df = gen.generate_normal(n=N_BASELINE, activity="resting")
    print(f"  Generated {len(baseline_df)} normal baseline readings.")

    # Labeled set: normal + all 4 anomaly types — used to train the classifier
    train_parts = [
        gen.generate_normal(n=80,  activity="resting"),
        gen.generate_anomaly("hypoxia",  n=20),
        gen.generate_anomaly("cardiac",  n=20),
        gen.generate_anomaly("stress",   n=20),
        gen.generate_anomaly("fatigue",  n=20),
    ]
    train_df = (pd.concat(train_parts, ignore_index=True)
                  .sample(frac=1, random_state=RANDOM_SEED)  # shuffle
                  .reset_index(drop=True))
    print(f"  Generated {len(train_df)} labeled classifier training samples.")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n[STEP 2] Preprocessing data...")
    preprocessor = DataPreprocessor()

    # fit_transform on baseline (fits the scaler)
    baseline_proc = preprocessor.fit_transform(baseline_df)

    # transform on training data (uses the already-fitted scaler)
    train_proc          = preprocessor.transform(train_df)
    train_proc["label"] = train_df["label"].values  # restore labels after transform

    feature_cols = [
        "heart_rate", "spo2", "gsr", "skin_temp",
        "hr_spo2_ratio", "temp_deviation", "gsr_hr_product"
    ]

    # ── Step 3: Train models ──────────────────────────────────────────────────
    print("\n[STEP 3] Training AI models...")
    detector = AnomalyDetector(contamination=0.08)
    detector.train(baseline_proc, feature_cols)

    classifier = AnomalyClassifier()
    classifier.train(train_proc, feature_cols)

    # ── Step 4: Generate test set ─────────────────────────────────────────────
    print("\n[STEP 4] Building test scenario (normal + injected anomalies)...")
    test_parts = [
        gen.generate_normal(n=N_TEST,         activity="resting"),
        gen.generate_anomaly("hypoxia",  n=N_ANOMALIES // 3),
        gen.generate_anomaly("cardiac",  n=N_ANOMALIES // 3),
        gen.generate_anomaly("stress",   n=N_ANOMALIES // 3),
    ]
    for df in test_parts:
        df["astronaut"] = gen.name  # tag all rows with astronaut name

    test_df = (pd.concat(test_parts, ignore_index=True)
                 .sample(frac=1, random_state=RANDOM_SEED)
                 .reset_index(drop=True))
    print(f"  Test set: {len(test_df)} readings  "
          f"({N_TEST} normal + {N_ANOMALIES} anomalies)")

    # ── Step 5: Run detection + classification ────────────────────────────────
    print("\n[STEP 5] Running detection and classification...")

    # Save raw (unscaled) vitals — used for alert display and charts
    raw_cols   = ["heart_rate", "spo2", "gsr", "skin_temp",
                  "label", "astronaut"]
    raw_values = test_df[raw_cols].copy()

    test_proc           = preprocessor.transform(test_df)
    test_proc["label"]  = raw_values["label"].values
    test_proc["astronaut"] = raw_values["astronaut"].values

    detected   = detector.predict(test_proc)
    classified = classifier.predict(detected)

    # Restore unscaled vitals for readable output
    for col in ["heart_rate", "spo2", "gsr", "skin_temp", "label", "astronaut"]:
        classified[col] = raw_values[col].values

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    classifier.evaluate(test_proc, feature_cols)

    # ── Step 7: Alerts ────────────────────────────────────────────────────────
    print("\n[STEP 6] Processing alerts...")
    alert_system  = AlertSystem()
    alerts_printed = 0

    for _, row in classified.iterrows():
        alert = alert_system.process(row)
        if alert is not None:
            if alerts_printed < 5:  # Show first 5 alerts on screen
                alert_system.print_alert(alert)
                alerts_printed += 1
            elif alerts_printed == 5:
                print("\n  ... (remaining alerts logged silently)\n")
                alerts_printed += 1  # prevent repeated message

    alert_system.summary()

    # ── Step 8: Charts ────────────────────────────────────────────────────────
    print("[STEP 7] Generating charts...")
    plot_vitals_with_anomalies(classified, astronaut_name=gen.name)
    plot_anomaly_distribution(classified)

    print("\n[DONE] AstroVital AI pipeline complete.")
    print(f"       Charts saved in:  ./{OUTPUT_DIR}/")
    print("="*60 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[STOPPED] Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        print("Check Python version (3.8+) and installed packages.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
