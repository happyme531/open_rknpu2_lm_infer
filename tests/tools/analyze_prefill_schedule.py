"""Summarize paired prefill wall time, process CPU and delayed NPU load."""
import json, re, statistics, sys
from pathlib import Path
out=Path(sys.argv[1] if len(sys.argv)>1 else 'llm_generated_docs/evidence/prefill_schedule_perf_20261025')
summary=[]
models=sys.argv[2:]or['qwen3','qwen35']
for name in models:
 root=out/name
 samples=[json.loads(line)for line in (root/'samples.jsonl').read_text().splitlines()]
 loads=[json.loads(line)for line in (root/'load.jsonl').read_text().splitlines()]
 for row in loads:
  row['npu']=[int(v)for v in re.findall(r'Core\d+:\s*(\d+)%',row['load'])]
  assert len(row['npu'])==3
 for row in samples:
  assert row['stats']['position']==row['prompt']
  assert row['stats']['dual_prefill_groups']==(row['prompt']//256 if row['lanes']==2 else 0)
  for key in ['tune_cache_misses','tune_calls','strategy_fallback_calls','weight_reload_calls']:
   assert row['stats'][key]==0, ('untuned or reloaded sample',name,key,row['stats'][key])
  start,end=row['start_ns']+2_050_000_000,row['end_ns'];weight=0;sums=[0.,0.,0.]
  for a,b in zip(loads,loads[1:]):
   width=max(0,min(end,b['time_ns'])-max(start,a['time_ns']))
   weight+=width
   for c in range(3):sums[c]+=width*a['npu'][c]
  row['npu_window_ns']=weight;row['npu_percent']=[v/weight for v in sums]if weight else None
  assert weight>=(end-start)*.95, 'incomplete NPU monitor coverage'
  compiler_ns=0
  for a,b in zip(loads,loads[1:]):
   width=max(0,min(row['end_ns'],b['time_ns'])-max(row['start_ns'],a['time_ns']))
   if not width:continue
   if a.get('compilers'):compiler_ns+=width
  row['compiler_overlap_ms']=compiler_ns/1e6
  # Aggregate jiffies across the full interval; averaging 50 ms ratios adds
  # quantization noise and biases the system/process CPU comparison.
  a=min(loads,key=lambda x:abs(x['time_ns']-row['start_ns']))
  b=min(loads,key=lambda x:abs(x['time_ns']-row['end_ns']))
  row['system_big_core_percent']=[]
  for c in range(4,8):
   x=a['cpu']['cpu'+str(c)];y=b['cpu']['cpu'+str(c)]
   total=sum(y)-sum(x);idle=y[3]+y[4]-x[3]-x[4]
   row['system_big_core_percent'].append(100*(total-idle)/total)
  row['process_cpu_percent']=row['cpu_seconds']/(row['wall_ms']/1000)*25
  row['other_cpu_percentage_points']=statistics.mean(row['system_big_core_percent'])-row['process_cpu_percent']
  assert row['other_cpu_percentage_points']<5, ('external CPU interference',name,row['prompt'],row['cycle'],row['order'],row['other_cpu_percentage_points'])
  clocks={}
  for load in loads:
   if row['start_ns']<=load['time_ns']<=row['end_ns']:
    for path,value in load.get('clocks',{}).items():clocks.setdefault(path,set()).add(value)
  row['clock_values']={k:sorted(v)for k,v in clocks.items()}
  for path,values in clocks.items():
   expected='2400000'if'cpufreq'in path else '2112000000'if'/dmc/'in path else '1000000000'
   assert values=={expected}, ('clock changed',path,values)
 for prompt in [512,4096]:
  modes={}
  for lanes in [1,2]:
   rows=[r for r in samples if r['prompt']==prompt and r['lanes']==lanes]
   assert len(rows)==4,(name,prompt,lanes,len(rows))
   wall=sum(r['wall_ms']for r in rows);cpu=sum(r['cpu_seconds']for r in rows);window=sum(r['npu_window_ns']for r in rows)
   mean=statistics.mean(r['wall_ms']for r in rows)
   assert not any(r['compiler_overlap_ms']for r in rows), 'external compiler interfered with timed sample'
   modes[lanes]={'samples_ms':[r['wall_ms']for r in rows],'mean_ms':mean,'stddev_ms':statistics.stdev(r['wall_ms']for r in rows),'tokens_per_second':1000*prompt/mean,'cpu_core_seconds':cpu/len(rows),'cpu_percent_of_four_big_cores':cpu/(wall/1000)*25,'npu_percent':[sum(r['npu_percent'][c]*r['npu_window_ns']for r in rows)/window for c in range(3)],'npu_window_seconds':window/1e9,'system_big_core_percent':[statistics.mean(r['system_big_core_percent'][c]for r in rows)for c in range(4)]}
  summary.append({'model':name,'prompt':prompt,'modes':modes,'throughput_gain_percent':100*(modes[2]['tokens_per_second']/modes[1]['tokens_per_second']-1)})
 report={'samples':samples,'metadata':{k:samples[-1]['stats'].get(k)for k in ['tune_cache','tune_cache_hits','tune_cache_misses','tune_calls','strategy_fallback_calls','weight_reload_calls','mapped_weight_bytes','weight_payload_bytes','lm_head_weight_bits']}}
 (root/'analyzed.json').write_text(json.dumps(report,indent=2)+'\n')
filename='summary.json'if set(models)=={'qwen3','qwen35'}else 'summary-'+'-'.join(models)+'.json'
(out/filename).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
