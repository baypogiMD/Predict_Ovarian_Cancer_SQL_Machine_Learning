DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS biomarkers;
DROP TABLE IF EXISTS hematology;
DROP TABLE IF EXISTS biochemistry;

CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    menopause INTEGER,
    type INTEGER CHECK (type IN (0,1))
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

CREATE TABLE hematology (
    patient_id INTEGER,
    wbc REAL,
    rbc REAL,
    hgb REAL,
    hct REAL,
    plt REAL,
    rdw REAL,
    neu_pct REAL,
    lym_pct REAL,
    mono_pct REAL,
    eos_pct REAL,
    baso_pct REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE biochemistry (
    patient_id INTEGER,
    alt REAL,
    ast REAL,
    alp REAL,
    bun REAL,
    crea REAL,
    glu REAL,
    alb REAL,
    tp REAL,
    ag_ratio REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
