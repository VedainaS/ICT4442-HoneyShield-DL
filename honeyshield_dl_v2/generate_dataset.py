"""
generate_dataset.py
--------------------
Generates a synthetic enterprise login-event dataset for HoneyShield-DL.

This is an ORIGINAL implementation (not copied from any external report/code).
It produces ~15,000 login events across ~800 simulated users, with a 92:8
normal-to-anomalous split across five attack categories, matching the scale
and taxonomy described in the project synopsis.

Output: data/raw_logs.csv
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_USERS = 800
N_EVENTS = 15000
ANOMALY_RATE = 0.08  # 8% anomalous, matching the HoneyShield-style split
ATTACK_TYPES = [
    "Brute Force",
    "Credential Stuffing",
    "Impossible Travel",
    "Device Spoofing",
    "Privilege Escalation",
]
SENSITIVE_RESOURCES = ["admin_panel", "payroll_system", "hr_database", "finance_console"]
NORMAL_RESOURCES = ["email", "file_share", "intranet_portal", "ticketing_system", "chat_app"]

DEVICE_TYPES = ["laptop", "desktop", "mobile", "tablet"]
OS_LIST = ["Windows", "macOS", "Linux", "iOS", "Android"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"]


def build_user_population(n_users=N_USERS):
    """Each user gets a consistent behavioural baseline."""
    users = []
    for uid in range(n_users):
        users.append({
            "user_id": f"user_{uid:05d}",
            "home_country": fake.country(),
            "home_city": fake.city(),
            "home_ip_prefix": ".".join(str(random.randint(1, 254)) for _ in range(3)),
            "device_id": fake.uuid4()[:8],
            "device_type": random.choice(DEVICE_TYPES),
            "os": random.choice(OS_LIST),
            "browser": random.choice(BROWSERS),
            "typical_login_hour": random.randint(7, 19),  # working hours baseline
        })
    return pd.DataFrame(users)


def random_timestamp(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_normal_event(user, start_date, end_date):
    ts = random_timestamp(start_date, end_date)
    # vary login hour slightly around the user's baseline
    hour_jitter = int(np.clip(np.random.normal(user.typical_login_hour, 1.5), 0, 23))
    ts = ts.replace(hour=hour_jitter)
    return {
        "timestamp": ts,
        "user_id": user.user_id,
        "country": user.home_country,
        "ip_address": user.home_ip_prefix + f".{random.randint(1, 254)}",
        "device_id": user.device_id,
        "device_type": user.device_type,
        "os": user.os,
        "browser": user.browser,
        "resource_accessed": random.choice(NORMAL_RESOURCES),
        "login_success": True,
        "failed_attempts": 0,
        "session_duration": max(30, np.random.normal(600, 200)),
        "attack_type": "Normal",
        "is_anomaly_ground_truth": 0,
    }


def generate_attack_event(user, attack_type, start_date, end_date):
    base = generate_normal_event(user, start_date, end_date)
    base["attack_type"] = attack_type
    base["is_anomaly_ground_truth"] = 1

    if attack_type == "Brute Force":
        base["failed_attempts"] = random.randint(5, 15)
        base["login_success"] = False
        base["session_duration"] = max(5, np.random.normal(20, 10))
        base["device_id"] = fake.uuid4()[:8]  # unfamiliar device

    elif attack_type == "Credential Stuffing":
        base["failed_attempts"] = random.randint(3, 8)
        base["login_success"] = random.random() < 0.2
        base["ip_address"] = fake.ipv4()
        base["device_id"] = fake.uuid4()[:8]

    elif attack_type == "Impossible Travel":
        base["country"] = fake.country()
        while base["country"] == user.home_country:
            base["country"] = fake.country()
        base["login_success"] = True

    elif attack_type == "Device Spoofing":
        base["device_id"] = fake.uuid4()[:8]
        base["device_type"] = random.choice(DEVICE_TYPES)
        base["os"] = random.choice(OS_LIST)
        base["browser"] = random.choice(BROWSERS)
        base["login_success"] = True

    elif attack_type == "Privilege Escalation":
        base["resource_accessed"] = random.choice(SENSITIVE_RESOURCES)
        base["login_success"] = True
        base["session_duration"] = max(60, np.random.normal(1800, 400))

    return base


def generate_dataset(n_events=N_EVENTS, anomaly_rate=ANOMALY_RATE):
    users_df = build_user_population()
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 8, 1)

    n_anomalous = int(n_events * anomaly_rate)
    n_normal = n_events - n_anomalous
    n_per_attack = n_anomalous // len(ATTACK_TYPES)

    records = []
    for _ in range(n_normal):
        user = users_df.sample(1).iloc[0]
        records.append(generate_normal_event(user, start_date, end_date))

    for attack_type in ATTACK_TYPES:
        for _ in range(n_per_attack):
            user = users_df.sample(1).iloc[0]
            records.append(generate_attack_event(user, attack_type, start_date, end_date))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/raw_logs.csv", index=False)
    print(f"Generated {len(df)} events.")
    print(df["attack_type"].value_counts())
    print(f"Saved to data/raw_logs.csv")
