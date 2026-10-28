-- Perfetto v58.2. Model leaf spans are exclusive; native child slices provide
-- operator detail and must not be added to their model parents.
SELECT id, name, ts, dur / 1e6 AS wall_ms, track_id
FROM slice WHERE name IN ('llm.prefill', 'llm.decode') ORDER BY ts;

SELECT p.name AS phase, s.name AS stage, count(*) AS calls,
       sum(s.dur) / 1e6 AS milliseconds,
       100.0 * sum(s.dur) / max(p.dur) AS percent
FROM slice p JOIN slice s ON s.track_id = p.track_id
 AND s.ts >= p.ts AND s.ts + s.dur <= p.ts + p.dur
WHERE p.name IN ('llm.prefill', 'llm.decode') AND p.dur > 0
 AND s.name GLOB 'ling.*' AND s.name != 'ling.layer' AND s.dur >= 0
GROUP BY p.name, s.name ORDER BY p.name, milliseconds DESC;

SELECT p.name AS phase, extract_arg(l.arg_set_id, 'debug.layer') AS layer,
       s.name AS stage, sum(s.dur) / 1e6 AS milliseconds
FROM slice p JOIN slice l ON l.track_id = p.track_id
 AND l.ts >= p.ts AND l.ts + l.dur <= p.ts + p.dur
 JOIN slice s ON s.track_id = l.track_id
 AND s.ts >= l.ts AND s.ts + s.dur <= l.ts + l.dur
WHERE p.name IN ('llm.prefill', 'llm.decode') AND l.name = 'ling.layer'
 AND s.name GLOB 'ling.*' AND s.name != 'ling.layer' AND s.dur >= 0
GROUP BY p.name, layer, s.name ORDER BY p.name, layer, milliseconds DESC;

SELECT name, severity, value FROM stats WHERE value != 0 AND severity != 'info';
