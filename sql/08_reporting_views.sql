DROP VIEW IF EXISTS v_reporting_summary;

CREATE VIEW v_reporting_summary AS
SELECT
    type,
    COUNT(*) AS patients,
    ROUND(AVG(age),1) AS avg_age,
    ROUND(AVG(ca125),1) AS avg_ca125,
    ROUND(AVG(he4),1) AS avg_he4,
    ROUND(AVG(nlr),2) AS avg_nlr
FROM model_dataset
GROUP BY type;
