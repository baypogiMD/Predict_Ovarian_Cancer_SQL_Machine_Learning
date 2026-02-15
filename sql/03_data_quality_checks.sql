-- Age plausibility
SELECT COUNT(*) AS invalid_age
FROM patients
WHERE age < 10 OR age > 100;

-- Missing key tumor markers
SELECT
    SUM(CASE WHEN ca125 IS NULL THEN 1 ELSE 0 END) AS missing_ca125,
    SUM(CASE WHEN he4 IS NULL THEN 1 ELSE 0 END) AS missing_he4
FROM biomarkers;

-- Class balance
SELECT type, COUNT(*) AS n
FROM patients
GROUP BY type;

-- Menopause consistency
SELECT menopause, COUNT(*) FROM patients GROUP BY menopause;
