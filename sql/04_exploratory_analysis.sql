SELECT
    p.type,
    AVG(p.age) AS mean_age,
    AVG(b.ca125) AS mean_ca125,
    AVG(b.he4) AS mean_he4,
    AVG(b.cea) AS mean_cea,
    AVG(h.neu_pct) AS mean_neutrophils,
    AVG(h.lym_pct) AS mean_lymphocytes
FROM patients p
JOIN biomarkers b USING(patient_id)
JOIN hematology h USING(patient_id)
GROUP BY p.type;
