"""Audit the expert I/O timeline and summarize decode overlap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def merged(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if end<=begin:
            continue
        if result and begin<=result[-1][1]:
            result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:
            result.append((begin,end))
    return result


def intersect(a,b):
    result=[];i=j=0;a=merged(a);b=merged(b)
    while i<len(a) and j<len(b):
        begin=max(a[i][0],b[j][0]);end=min(a[i][1],b[j][1])
        if begin<end:
            result.append((begin,end))
        if a[i][1]<b[j][1]:
            i+=1
        else:
            j+=1
    return result


def duration(intervals):
    return sum(end-begin for begin,end in merged(intervals))/1000


def audit(events):
    has_issued=any(e['name']=='read issued' for e in events)
    actions=[]
    for index,e in enumerate(events):
        actions.append((e['ts'],index,'start',e))
        if e['ph']=='X':
            actions.append((e['ts']+e['dur'],index,'end',e))
    demand=set();ready=set();taken=set();slots={};violations=[];reads=0;imports=0;calls=0;slot_checks=0
    for timestamp,index,phase,e in sorted(actions):
        name=e['name'];a=e.get('args',{});key=(a.get('layer'),a.get('expert'))
        if phase=='start':
            if name=='reset':
                demand.clear();ready.clear();taken.clear()
            elif name=='enqueue demand':
                demand.add(key)
            elif name=='pread':
                reads+=1;taken.discard(key)
                if not has_issued and a['prefetch'] and demand:
                    violations.append({'kind':'prefetch_before_queued_demand','at_us':timestamp,'read':a,'queued':sorted(demand)})
                if not has_issued:
                    demand.discard(key)
            elif name=='read issued':
                if a['prefetch'] and a['queued_demands']:
                    violations.append({'kind':'prefetch_before_queued_demand','at_us':timestamp,'read':a})
                demand.discard(key)
            elif name=='read ready':
                ready.add(key)
            elif name=='read discarded':
                ready.discard(key)
            elif name=='cancel prefetch':
                ready.discard(key)
            elif name=='import':
                imports+=1
                if key not in taken:
                    violations.append({'kind':'import_before_read_handoff','at_us':timestamp,'import':a})
            elif name=='NPU experts':
                calls+=1
                bank=a['bank'];layer=a['layer']
                present={v for k,v in slots.items() if k[0]==bank and (bank=='cold' or k[1]==layer)}
                for expert in set(a['experts']):
                    if (layer,expert) not in present:
                        violations.append({'kind':'compute_before_expert_import','at_us':timestamp,'layer':layer,'expert':expert,'bank':bank})
                for expert,slot in zip(a['experts'],a.get('slots',[])):
                    slot_checks+=1
                    slot_key=(bank,layer if bank=='hot' else -1,slot)
                    if slots.get(slot_key)!=(layer,expert):
                        violations.append({'kind':'wrong_physical_slot','at_us':timestamp,'layer':layer,'expert':expert,'bank':bank,'slot':slot})
        else:
            if name=='wait record':
                if key not in ready:
                    violations.append({'kind':'handoff_before_read_ready','at_us':timestamp,'layer':key[0],'expert':key[1]})
                ready.discard(key);taken.add(key)
            elif name=='import':
                slot=(a['bank'],a['layer'] if a['bank']=='hot' else -1,a['slot'])
                slots[slot]=key
    return {'read_calls':reads,'imports':imports,'expert_calls':calls,'physical_slot_checks':slot_checks,'violations':violations}


def summarize(path):
    events=json.loads(path.read_text())['traceEvents']
    spans=[e for e in events if e['ph']=='X']
    begins={e['args']['position']:e for e in events if e['name']=='forward begin'}
    windows=[]
    for e in events:
        if e['name']=='forward end':
            begin=begins[e['args']['position']]
            windows.append((begin,e))
    decode=[(a['ts'],b['ts']) for a,b in windows if a['args']['rows']==1]
    ranges=lambda name:[(e['ts'],e['ts']+e['dur']) for e in spans if e['name']==name]
    io=intersect(ranges('pread'),decode);wait=intersect(ranges('wait record'),decode)
    compute=intersect(ranges('NPU experts'),decode)
    other_speculative=[]
    for w in spans:
        if w['name']!='wait record':
            continue
        for read in spans:
            if read['name']=='pread' and read['args']['prefetch'] and read['args']['layer']!=w['args']['layer']:
                other_speculative.extend(intersect([(w['ts'],w['ts']+w['dur'])],[(read['ts'],read['ts']+read['dur'])]))
    report={'trace':str(path),'audit':audit(events),'decode':{
        'tokens':len(decode),'wall_ms':duration(decode),'io_ms':duration(io),'wait_ms':duration(wait),
        'expert_call_ms':duration(compute),'io_expert_overlap_ms':duration(intersect(io,compute)),
        'io_nonwaiting_main_overlap_ms':duration(io)-duration(intersect(io,wait)),
        'wait_on_other_layer_prefetch_ms':duration(intersect(other_speculative,decode)),
    }}
    predictions=[e for e in events if e['name']=='LES prediction']
    leads=[]
    for e in events:
        if e['name']=='actual routes' and e['args']['rows']==1 and e['args']['layer']>0:
            matches=[p for p in predictions if p['args']['layer']==e['args']['layer'] and p['ts']<=e['ts']]
            if not matches:
                continue
            prediction=max(matches,key=lambda p:p['ts']);a=e['args']
            leads.append({'position':a['position'],'layer':a['layer'],'lead_ms':(e['ts']-prediction['ts'])/1000,
                'routes':a['experts'],'cold':a['cold'],'predicted':prediction['args']['experts']})
    report['prediction_leads']=leads
    return report,events,windows


def plot(events,windows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    begin,end=next((a,b) for a,b in windows if a['args']['rows']==1)
    origin=begin['ts'];limit=min(end['ts'],origin+180000)
    colors={'pread':'#2878b5','wait record':'#e68632','NPU experts':'#3a923a','import':'#9566b0'}
    rows={'pread':3,'wait record':0,'NPU experts':1,'import':2}
    fig,ax=plt.subplots(figsize=(13,3.6))
    for e in events:
        if e['ph']!='X' or e['name'] not in rows:
            continue
        first=max(origin,e['ts']);last=min(limit,e['ts']+e['dur'])
        if last<=first:
            continue
        name=e['name'];a=e['args'];color=colors[name]
        if name=='pread' and not a['prefetch']:
            color='#74add1'
        ax.broken_barh([((first-origin)/1000,(last-first)/1000)],(rows[name]-.3,.6),facecolors=color)
        if last-first>2200:
            ax.text(((first+last)/2-origin)/1000,rows[name],f"L{a['layer']}",ha='center',va='center',fontsize=7)
    ax.set_yticks([0,1,2,3],['Wait for record','Expert call','Import + sync','I/O: dark=prefetch'])
    ax.set_xlabel('ms from first decode call');ax.set_xlim(0,(limit-origin)/1000)
    ax.grid(axis='x',alpha=.25);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('trace',type=Path)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--plot',type=Path)
    args=parser.parse_args();report,events,windows=summarize(args.trace)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if args.plot:
        plot(events,windows,args.plot)
    print(json.dumps({'audit':{k:v for k,v in report['audit'].items() if k!='violations'},
        'violations':len(report['audit']['violations']),'decode':report['decode']},indent=2))


if __name__=='__main__':
    main()
