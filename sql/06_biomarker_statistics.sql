SELECT
    type,
    COUNT(*) AS n,
    AVG(ca125) AS mean_ca125,
    MIN(ca125) AS min_ca125,
    MAX(ca125) AS max_ca125,
    AVG(he4) AS mean_he4
FROM v_clinical_features
GROUP BY type;
