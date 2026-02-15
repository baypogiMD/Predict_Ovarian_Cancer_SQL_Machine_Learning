DROP VIEW IF EXISTS v_clinical_features;

CREATE VIEW v_clinical_features AS
SELECT
    p.patient_id,
    p.age,
    p.menopause,
    b.ca125,
    b.he4,
    b.cea,
    b.afp,
    b.ca199,
    b.ca724,
    h.wbc,
    h.hgb,
    h.plt,
    h.rdw,
    h.neu_pct,
    h.lym_pct,
    (h.neu_pct / NULLIF(h.lym_pct, 0)) AS nlr,
    bi.alt,
    bi.ast,
    bi.bun,
    bi.crea,
    bi.glu,
    p.type
FROM patients p
JOIN biomarkers b USING(patient_id)
JOIN hematology h USING(patient_id)
JOIN biochemistry bi USING(patient_id);
