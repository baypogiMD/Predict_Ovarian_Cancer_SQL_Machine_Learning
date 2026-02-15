DROP TABLE IF EXISTS model_dataset;

CREATE TABLE model_dataset AS
SELECT
    patient_id,
    age,
    menopause,
    ca125,
    he4,
    cea,
    afp,
    ca199,
    ca724,
    wbc,
    hgb,
    plt,
    rdw,
    neu_pct,
    lym_pct,
    nlr,
    alt,
    ast,
    bun,
    crea,
    glu,
    type
FROM v_clinical_features
WHERE ca125 IS NOT NULL
  AND he4 IS NOT NULL;
