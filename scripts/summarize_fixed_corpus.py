"""Average repeated judgments and schema samples within each newly generated corpus."""
import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpora',nargs='+',type=Path);a=p.parse_args()
all_scores={'hierarchical':[],'strict':[]}
for d in a.corpora:
    per={'hierarchical':[],'strict':[]}
    cfg=json.loads((d/'config.json').read_text())
    for i in range(1,cfg['anchor_samples']+1):
        vals={k:[] for k in per}
        for j in range(1,cfg['judge_reps']+1):
            ev=d/f's{i}/evaluation/rep{j}'
            vals['hierarchical'].append(json.loads((ev/'hierarchical/metrics_fine.json').read_text())['fine_f1'])
            with (ev/'strict/metrics.csv').open() as f:
                row=list(csv.reader(f))[1];vals['strict'].append(float(row[11]))
        for k in per:per[k].append(statistics.mean(vals[k]))
    result={k:statistics.mean(v) for k,v in per.items()}
    print(d.name,json.dumps(result))
    for k in result:all_scores[k].append(result[k])
print('Across-corpus mean:',json.dumps({k:statistics.mean(v) for k,v in all_scores.items()}))
