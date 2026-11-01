"""User-disjoint splits; training-only preprocessing; fully normal train/val windows.
Test labels describe the last event in each window. Unknown categories map to zero.
"""
import numpy as np
import pandas as pd
import os

SEQ_LEN = 10  # fixed-length window per user (N=10 events)


def frequency_encode(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq).astype(float)


def build_features(df, fit_df=None):
    if fit_df is None:
        raise ValueError("Training events are required to fit preprocessing")
    fit_df = fit_df.copy()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["login_hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday.isin([5, 6]).astype(int)

    # Behavioural outcome flags
    df["login_success_flag"] = df["login_success"].astype(int)
    df["is_failed_login"] = (df["failed_attempts"] > 0).astype(int)
    df["is_sensitive_resource"] = df["resource_accessed"].isin(
        ["admin_panel", "payroll_system", "hr_database", "finance_console"]
    ).astype(int)
    df["unusual_login_hour"] = (~df["login_hour"].between(6, 22)).astype(int)

    # Statistical deviation feature
    df["session_duration_zscore"] = (
        (df["session_duration"] - fit_df["session_duration"].mean())
        / max(float(fit_df["session_duration"].std()), 1e-8)
    )

    # Frequency maps fitted exclusively on normal training-user events.
    for col in ["country", "resource_accessed", "browser", "device_type", "ip_address"]:
        freq = fit_df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0.0).astype(float)

    feature_cols = [
        "login_success_flag", "is_failed_login", "is_sensitive_resource",
        "unusual_login_hour", "session_duration_zscore",
        "country_freq", "resource_accessed_freq", "browser_freq",
        "device_type_freq", "ip_address_freq", "failed_attempts",
    ]
    return df, feature_cols


def build_sequences(df, feature_cols, seq_len=SEQ_LEN):
    X_list, y_list, attack_list = [], [], []
    num_features = len(feature_cols)

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feats = group[feature_cols].values.astype(np.float32)
        labels = group["is_anomaly_ground_truth"].values
        attacks = group["attack_type"].values

        n = len(group)
        for end_idx in range(1, n + 1):
            start_idx = max(0, end_idx - seq_len)
            window_feats = feats[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]
            window_attacks = attacks[start_idx:end_idx]

            pad_len = seq_len - len(window_feats)
            if pad_len > 0:
                pad = np.zeros((pad_len, num_features), dtype=np.float32)
                window_feats = np.vstack([pad, window_feats])

            # Label corresponds to the LAST event in the window (the event being
            # classified); earlier events in the window provide temporal context only.
            # This keeps the sequence-level anomaly rate consistent with the
            # original per-event rate (~8%), unlike an "any event in window"
            # rule which would inflate it.
            seq_label = int(window_labels[-1])
            dominant_attack = window_attacks[-1]

            X_list.append(window_feats)
            y_list.append(seq_label)
            attack_list.append(dominant_attack)

    X = np.stack(X_list)
    y = np.array(y_list)
    attacks = np.array(attack_list)
    return X, y, attacks


def prepare_data():
    import json
    from sklearn.model_selection import train_test_split
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/raw_logs.csv")
    users = sorted(df.user_id.unique())
    trainval, test = train_test_split(users, test_size=0.2, random_state=42)
    train, val = train_test_split(trainval, test_size=0.15, random_state=42)
    fit = df[df.user_id.isin(train) & (df.is_anomaly_ground_truth == 0)]
    arrays = []
    manifest = {}
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        subset, cols = build_features(df[df.user_id.isin(ids)], fit)
        X, y, attacks = build_sequences(subset, cols)
        clean = []
        for _, group in subset.groupby("user_id"):
            labels = group.sort_values("timestamp").is_anomaly_ground_truth.to_numpy()
            clean.extend(not labels[max(0,i-9):i+1].any() for i in range(len(labels)))
        clean = np.asarray(clean)
        keep = clean if name != "test" else np.ones(len(y), dtype=bool)
        arrays.append((X[keep], y[keep], attacks[keep], np.full(keep.sum(), name)))
        manifest[name] = {"users": sorted(ids), "sequences": int(keep.sum()), "anomalies": int(y[keep].sum())}
    X,y,a,split = [np.concatenate([row[i] for row in arrays]) for i in range(4)]
    for name, arr in [("X",X),("y",y),("attack",a),("split",split)]:
        np.save("data/sequences_"+name+".npy",arr)
    preprocessing = {"duration_mean": float(fit.session_duration.mean()), "duration_std": float(fit.session_duration.std()),
        "frequencies": {col: fit[col].value_counts(normalize=True).to_dict() for col in ["country","resource_accessed","browser","device_type","ip_address"]}, "features": cols}
    with open("data/preprocessing.json","w") as f: json.dump(preprocessing,f,indent=2)
    with open("data/split_manifest.json","w") as f: json.dump(manifest,f,indent=2)
    print({k: {a:b for a,b in v.items() if a != "users"} for k,v in manifest.items()})

if __name__ == "__main__":
    prepare_data()
