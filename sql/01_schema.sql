CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    menopause INTEGER,
    type INTEGER -- 0=cancer, 1=benign
);

CREATE TABLE biomarkers (
    patient_id INTEGER,
    ca125 REAL,
    he4 REAL,
    cea REAL,
    afp REAL,
    ca199 REAL,
    ca724 REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE labs (
    patient_id INTEGER,
    wbc REAL,
    rbc REAL,
    hgb REAL,
    plt REAL,
    neu_pct REAL,
    lym_pct REAL,
    mono_pct REAL,
    rdw REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
