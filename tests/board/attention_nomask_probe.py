#!/usr/bin/env python3
"""Complete no-mask MHA calls: cache load, query blocks, sync and output collection."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np


class Config(C.Structure):
    _fields_ = [(x,C.c_int) for x in ('query_heads','kv_heads','head_dim','max_query_rows','kv_tile','initial_capacity','max_capacity')] + [(x,C.c_uint32) for x in ('core_mask','timeout_ms','flags')] + [('exp_shift',C.c_float),('mask_mode',C.c_int),('softmax_scale',C.c_float)]


class Sizes(C.Structure):
    _fields_ = [(x,C.c_uint64) for x in ('key_bytes','value_bytes','tail_bytes','query_bytes','score_bytes','mask_bytes','partial_bytes','output_bytes')] + [(x,C.c_int) for x in ('compute_length','physical_rows','tasks')] + [('active_core_mask',C.c_uint32)]


class Timings(C.Structure):
    _fields_ = [(x,C.c_double) for x in ('reserve_us','kv_update_us','prepare_us','tail_stage_us','mask_update_us','query_pack_us','submit_us','hardware_us','output_sync_us','finish_us','total_us')] + [('compute_length',C.c_int),('tasks',C.c_int)]


def pointer(a):
    return C.c_void_p(a.ctypes.data)


def error(reference, actual):
    a,b=reference.astype(np.float64),actual.astype(np.float64)
    assert a.shape==b.shape and np.isfinite(b).all()
    return dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),max_abs=float(np.abs(a-b).max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,nargs='+',default=[256,384,1024])
    p.add_argument('--query-rows',type=int,default=128)
    p.add_argument('--tile',type=int,default=512)
    p.add_argument('--core-mask',type=int,default=7)
    p.add_argument('--storage',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--value-scale',type=float,default=1.)
    p.add_argument('--loops',type=int,default=7)
    p.add_argument('--warmup',type=int,default=2)
    args=p.parse_args()
    lib=C.CDLL(str(args.library.resolve()),mode=C.RTLD_LOCAL)
    P=C.c_void_p; I=C.c_int
    def bind(name,types,result=I):
        f=getattr(lib,name);f.argtypes=types;f.restype=result;return f
    open_=bind('rk_npu_open',[C.c_char_p],P)
    close=bind('rk_npu_close',[P],None)
    domain_create=bind('rk_npu_iommu_domain_create',[P,C.c_uint32],P)
    domain_free=bind('rk_npu_iommu_domain_free',[P],None)
    init=bind('rk_npu_attention_f16_config_init',[C.POINTER(Config)],None)
    create=bind('rk_npu_attention_f16_cache_create',[P,C.POINTER(Config),C.POINTER(P)])
    free=bind('rk_npu_attention_f16_cache_free',[P],None)
    prepare=bind('rk_npu_attention_f16_prepare',[P,C.POINTER(Config),C.POINTER(P)])
    work_free=bind('rk_npu_attention_f16_workspace_free',[P],None)
    load=bind('rk_npu_attention_f16_cache_load',[P,P,P,P,I])
    query=bind('rk_npu_attention_f16_query',[C.POINTER(Config),I,I,I,C.POINTER(Sizes)])
    run=bind('rk_npu_attention_f16_run_half' if args.storage=='fp16' else 'rk_npu_attention_f16_run',[P,P,P,P,I,I,P,C.POINTER(Timings),P])
    strerror=bind('rk_npu_strerror',[I],C.c_char_p)
    detail=bind("attention_probe_read",[C.POINTER(C.c_double),I],None) if hasattr(lib,"attention_probe_read") else None
    counters=(C.c_double*5)()
    def check(rc):
        if rc:raise RuntimeError(strerror(rc).decode())
    ctx=open_(None);assert ctx
    domain=domain_create(ctx,14);assert domain
    results={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for length in args.length:
            capacity=(length+31)//32*32
            cfg=Config();init(C.byref(cfg))
            cfg.query_heads=cfg.kv_heads=16;cfg.head_dim=64
            cfg.max_query_rows=args.query_rows;cfg.kv_tile=args.tile
            cfg.initial_capacity=cfg.max_capacity=capacity
            cfg.core_mask=args.core_mask;cfg.flags=2;cfg.mask_mode=1
            sizes=Sizes();check(query(C.byref(cfg),capacity,min(length,args.query_rows),length,C.byref(sizes)))
            cache=P();work=P()
            check(create(domain,C.byref(cfg),C.byref(cache)))
            check(prepare(domain,C.byref(cfg),C.byref(work)))
            try:
                rng=np.random.default_rng(20261018+length)
                q,k,v=[rng.normal(0,sigma,(16,length,64)).astype(np.float16) for sigma in (.7,.7,.5*args.value_scale)]
                output=np.empty_like(q,dtype=np.float16 if args.storage=='fp16' else np.float32)
                chunks=[]
                for start in range(0,length,args.query_rows):
                    count=min(args.query_rows,length-start)
                    # Compact host ABI, matching the caller's head-major chunk staging.
                    qc=np.ascontiguousarray(q[:,start:start+count])
                    guard=np.full(16*count*64+16,12345,dtype=output.dtype)
                    chunks.append((start,count,qc,guard))
                samples=[];first=None
                for iteration in range(args.warmup+args.loops):
                    times={x:0. for x,_ in Timings._fields_}
                    if detail:detail(counters,1)
                    begin=time.perf_counter_ns()
                    check(load(ctx,cache,pointer(k),pointer(v),length))
                    loaded=time.perf_counter_ns()
                    for start,count,qc,guard in chunks:
                        t=Timings()
                        check(run(ctx,work,cache,pointer(qc),count,start,pointer(guard[8:-8]),C.byref(t),None))
                        output[:,start:start+count]=guard[8:-8].reshape(16,count,64)
                        for key,_ in Timings._fields_:
                            times[key]=max(times[key],getattr(t,key)) if key=="compute_length" else times[key]+getattr(t,key)
                    done=time.perf_counter_ns()
                    for _,_,_,guard in chunks:assert np.all(guard[:8]==12345) and np.all(guard[-8:]==12345)
                    if first is None:first=output.copy()
                    else:assert np.array_equal(first.view(np.uint16 if args.storage=='fp16' else np.uint32),output.view(np.uint16 if args.storage=='fp16' else np.uint32))
                    if iteration>=args.warmup:
                        samples.append(dict(wall_us=(done-begin)/1000,load_us=(loaded-begin)/1000,native=times))
                        if detail:
                            detail(counters,0)
                            samples[-1]['detail']=dict(zip(('qk_us','score_from_us','center_us','score_to_us','lut_pv_us'),counters))
                scores=(q.astype(np.float32)@k.astype(np.float32).transpose(0,2,1))/8
                scores-=scores.max(axis=-1,keepdims=True)
                probabilities=np.exp(scores);probabilities/=probabilities.sum(axis=-1,keepdims=True)
                reference=probabilities@v.astype(np.float32)
                precision=error(reference,output)
                assert precision['relative_l2']<.01,precision
                output.astype('<f4').tofile(args.output.with_name(args.output.stem+f'-{length}.f32'))
                results[str(length)]=dict(median_us=float(np.median([s['wall_us'] for s in samples])),samples=samples,
                    precision=precision,repeatable=True,physical_rows=sizes.physical_rows,
                    queried_tasks=sizes.tasks,score_bytes=sizes.score_bytes,
                    max_value=float(np.abs(v.astype(np.float32)).max()))
                args.output.write_text(json.dumps(dict(library=str(args.library),query_rows=args.query_rows,
                    tile=args.tile,core_mask=args.core_mask,storage=args.storage,value_scale=args.value_scale,results=results),indent=2)+'\n')
                print(length,results[str(length)]['median_us'],precision,flush=True)
            finally:
                work_free(work);free(cache)
    finally:
        domain_free(domain);close(ctx)


if __name__=='__main__':main()
