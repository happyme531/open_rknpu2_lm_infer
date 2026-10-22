-- Materialize a timestamp index for the per-call interval joins.
CREATE PERFETTO TABLE ling_profile_slices AS
 SELECT id,name,ts,dur,track_id,arg_set_id FROM slice;
CREATE PERFETTO INDEX ling_profile_ts ON ling_profile_slices(ts);

-- Perfetto v58.2. CPU elapsed spans, not summed worker-core CPU cycles.
SELECT name, track_id, ts, dur, dur / 1e6 AS wall_ms
FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode') ORDER BY ts;

WITH phases AS MATERIALIZED (
  SELECT name,track_id,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
)
SELECT p.name AS phase,s.name AS stage,count(*) AS calls,
       sum(s.dur)/1e6 AS ms,100.0*sum(s.dur)/max(p.dur) AS percent
FROM phases p CROSS JOIN ling_profile_slices s ON s.track_id=p.track_id
 AND s.ts>=p.ts AND s.ts+s.dur<=p.ts+p.dur
WHERE s.dur>=0 AND s.name GLOB 'ling.*' AND s.name!='ling.layer'
GROUP BY p.name,s.name ORDER BY p.name,ms DESC;

-- moe.timings exports existing ABI stage measurements in microseconds.
-- Its own tiny slice duration is only the cost of emitting the event.
WITH phases AS MATERIALIZED (
  SELECT name,track_id,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
)
SELECT p.name AS phase,count(*) AS calls,
 sum(extract_arg(s.arg_set_id,'debug.input_us'))/1000 AS input_ms,
 sum(extract_arg(s.arg_set_id,'debug.schedule_us'))/1000 AS schedule_ms,
 sum(extract_arg(s.arg_set_id,'debug.build_us'))/1000 AS build_ms,
 sum(extract_arg(s.arg_set_id,'debug.sync_us'))/1000 AS sync_ms,
 sum(extract_arg(s.arg_set_id,'debug.gather_us'))/1000 AS gather_ms,
 sum(extract_arg(s.arg_set_id,'debug.gate_us'))/1000 AS gate_ms,
 sum(extract_arg(s.arg_set_id,'debug.middle_cpu_us'))/1000 AS middle_cpu_ms,
 sum(extract_arg(s.arg_set_id,'debug.middle_npu_us'))/1000 AS middle_npu_ms,
 sum(extract_arg(s.arg_set_id,'debug.lut_load_us'))/1000 AS lut_load_ms,
 sum(extract_arg(s.arg_set_id,'debug.combine_us'))/1000 AS combine_ms,
 sum(extract_arg(s.arg_set_id,'debug.down_us'))/1000 AS down_ms,
 sum(extract_arg(s.arg_set_id,'debug.total_us'))/1000 AS total_ms
FROM phases p CROSS JOIN ling_profile_slices s ON s.track_id=p.track_id
 AND s.ts>=p.ts AND s.ts+s.dur<=p.ts+p.dur
WHERE s.name='moe.timings' GROUP BY p.name;

-- All tracks: W8 pack/reduce runs on worker tracks. Do not include submit/wait
-- as CPU arithmetic, or sum nested swiglu_epilogue together with wave_reduce.
WITH phases AS MATERIALIZED (
  SELECT name,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
)
SELECT p.name AS phase,s.name AS stage,count(*) AS calls,sum(s.dur)/1e6 AS ms
FROM phases p CROSS JOIN ling_profile_slices s ON s.ts>=p.ts AND s.ts+s.dur<=p.ts+p.dur
WHERE s.dur>=0 AND s.name IN (
 'w4a8.activation_scan','i4.input_pack','i4.wave_reduce',
 'w4a8.chain_pack','w4a8.chain_reduce',
 'w8.wave_pack','w8.chain_pack','w8.output_reduce_dequant',
 'w8.partial_accumulate','w8.chain_output_copy',
 'attention.query_pack','attention.cache_pack','attention.center_scores',
 'attention.stable_softmax','attention.output_finish')
GROUP BY p.name,s.name ORDER BY p.name,ms DESC;


-- Model stages by layer; summed over four prefill chunks or 128 decode steps.
WITH phases AS MATERIALIZED (
 SELECT name,track_id,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
), layers AS MATERIALIZED (SELECT * FROM ling_profile_slices WHERE name='ling.layer')
SELECT p.name AS phase,extract_arg(l.arg_set_id,'debug.layer') AS layer,
 s.name AS stage,count(*) AS calls,sum(s.dur)/1e6 AS ms
FROM phases p CROSS JOIN layers l ON l.track_id=p.track_id AND l.ts>=p.ts AND l.ts+l.dur<=p.ts+p.dur
 CROSS JOIN ling_profile_slices s ON s.track_id=l.track_id AND s.ts>=l.ts AND s.ts<=l.ts+l.dur AND s.ts+s.dur<=l.ts+l.dur
WHERE s.dur>=0 AND s.name GLOB 'ling.*' AND s.name!='ling.layer'
GROUP BY phase,layer,stage ORDER BY phase,layer,stage;

-- All native names are retained separately; these nested values are NOT additive.
WITH phases AS MATERIALIZED (
 SELECT name,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
)
SELECT p.name AS phase,s.name AS stage,count(*) AS calls,sum(s.dur)/1e6 AS ms
FROM phases p CROSS JOIN ling_profile_slices s ON s.ts>=p.ts AND s.ts+s.dur<=p.ts+p.dur
WHERE s.dur>=0 AND s.name NOT GLOB 'ling.*' AND s.name NOT GLOB 'llm.*'
GROUP BY phase,stage ORDER BY phase,ms DESC;

-- All tracks, attributed to synchronous model calls by enclosing time interval.
WITH phases AS MATERIALIZED (
 SELECT name,track_id,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
), leaves AS MATERIALIZED (
 SELECT * FROM ling_profile_slices WHERE name GLOB 'ling.*' AND name!='ling.layer' AND dur>=0
)
SELECT p.name AS phase,l.name AS model_stage,s.name AS native_stage,
 count(*) AS calls,sum(s.dur)/1e6 AS ms
FROM phases p CROSS JOIN leaves l ON l.track_id=p.track_id AND l.ts>=p.ts AND l.ts+l.dur<=p.ts+p.dur
 CROSS JOIN ling_profile_slices s ON s.ts>=l.ts AND s.ts<=l.ts+l.dur AND s.ts+s.dur<=l.ts+l.dur
WHERE s.dur>=0 AND s.name NOT GLOB 'ling.*' AND s.name NOT GLOB 'llm.*'
GROUP BY phase,model_stage,native_stage ORDER BY phase,model_stage,ms DESC;

WITH phases AS MATERIALIZED (
 SELECT name,track_id,ts,dur FROM ling_profile_slices WHERE name IN ('llm.prefill','llm.decode')
), leaves AS MATERIALIZED (
 SELECT * FROM ling_profile_slices WHERE name GLOB 'ling.*' AND name!='ling.layer' AND dur>=0
)
SELECT p.name AS phase,l.name AS model_stage,s.name AS op,
 extract_arg(s.arg_set_id,'debug.M') AS m,extract_arg(s.arg_set_id,'debug.K') AS k,
 extract_arg(s.arg_set_id,'debug.N') AS n,count(*) AS calls,sum(s.dur)/1e6 AS ms
FROM phases p CROSS JOIN leaves l ON l.track_id=p.track_id AND l.ts>=p.ts AND l.ts+l.dur<=p.ts+p.dur
 CROSS JOIN ling_profile_slices s ON s.ts>=l.ts AND s.ts<=l.ts+l.dur AND s.ts+s.dur<=l.ts+l.dur
WHERE s.name IN ('w8.run','w4a8.run')
GROUP BY phase,model_stage,op,m,k,n ORDER BY phase,model_stage;

SELECT name,severity,value FROM stats WHERE value!=0 AND severity!='info';
