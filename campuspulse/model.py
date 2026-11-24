import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans

FEATURES = ["attendance", "internal_marks", "assignment_completion", "previous_result"]

class CampusModelSuite:
    """Demonstration model suite trained on generated synthetic examples only."""
    def __init__(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.uniform(35, 100, 3000), rng.uniform(0, 30, 3000),
            rng.uniform(0, 100, 3000), rng.uniform(20, 100, 3000)
        ])
        a, m, c, prev = X.T
        risk_points = ((a < 65).astype(int) + (m < 15).astype(int) +
                       (c < 60).astype(int) + (prev < 50).astype(int))
        y = (risk_points >= 2).astype(int)
        self.rf = RandomForestClassifier(n_estimators=160, max_depth=8, random_state=42, class_weight="balanced")
        self.hgb = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, random_state=42)
        self.rf.fit(X, y)
        self.hgb.fit(X, y)
        self.anomaly = IsolationForest(contamination=0.12, random_state=42)
        self.anomaly.fit(X)
        self.cluster = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.cluster.fit(X)

    def predict(self, row):
        x = np.array([[float(row[k]) for k in FEATURES]])
        rf_score = float(self.rf.predict_proba(x)[0][1])
        hgb_score = float(self.hgb.predict_proba(x)[0][1])
        score = round(((rf_score + hgb_score) / 2) * 100)
        level = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
        anomaly = bool(self.anomaly.predict(x)[0] == -1)
        segment = int(self.cluster.predict(x)[0]) + 1
        return {
            "score": score, "level": level,
            "model_scores": {"Random Forest": round(rf_score * 100), "Gradient Boosting": round(hgb_score * 100)},
            "anomaly": anomaly, "segment": segment
        }

# Backwards-compatible name for code that used the starter class.
RiskModel = CampusModelSuite
