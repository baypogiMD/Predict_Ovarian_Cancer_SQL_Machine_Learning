DROP TABLE IF EXISTS raw_data;

CREATE TABLE raw_data (
    patient_id INTEGER,
    age INTEGER,
    menopause INTEGER,
    type INTEGER,
    ca125 REAL,
    he4 REAL,
    cea REAL,
    afp REAL,
    ca199 REAL,
    ca724 REAL,
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
    alt REAL,
    ast REAL,
    alp REAL,
    bun REAL,
    crea REAL,
    glu REAL,
    alb REAL,
    tp REAL,
    ag_ratio REAL
);

-- Insert normalized data
INSERT INTO patients
SELECT patient_id, age, menopause, type FROM raw_data;

INSERT INTO biomarkers
SELECT patient_id, ca125, he4, cea, afp, ca199, ca724 FROM raw_data;

INSERT INTO hematology
SELECT patient_id, wbc, rbc, hgb, hct, plt, rdw,
       neu_pct, lym_pct, mono_pct, eos_pct, baso_pct
FROM raw_data;

INSERT INTO biochemistry
SELECT patient_id, alt, ast, alp, bun, crea, glu, alb, tp, ag_ratio
FROM raw_data;
