"""Run construction and two judges on a user-supplied, fixed PDF collection.
Default: validate inputs and print configuration only. --execute enables paid API calls.
This entry point reruns the method, not the historical cross-run replay.
"""
import argparse, json, os, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TIER = {'COF':'Covalent Organic Frameworks (COFs)', 'MOF':'Metal-organic frameworks (MOFs)', 'ZIF':'Zeolitic imidazolate frameworks (ZIFs)'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf-dir',type=Path,required=True)
    p.add_argument('--subdomain',choices=TIER,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--anchor-samples',type=int,default=6)
    p.add_argument('--judge-reps',type=int,default=2)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    pdfs=sorted(a.pdf_dir.resolve().glob('*.pdf'))
    if not pdfs:p.error('No PDF files found. Supply the intended fixed collection.')
    if a.anchor_samples<1 or a.judge_reps<1:p.error('Repetition counts must be positive')
    out=a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty; use a new directory to prevent mixed runs.')
    cfg={'pdfs':[str(f) for f in pdfs], 'subdomain':a.subdomain, 'anchor_samples':a.anchor_samples, 'judge_reps':a.judge_reps,
         'fields':8,'max_text_chars':12000,'mode':'fixed collection, fresh paper proposals; no historical replay'}
    print(json.dumps(cfg,indent=2))
    if not a.execute:
        print('Validation only. No API calls. Add --execute to run (paid).');return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    os.environ['MEMOSCHEMA_FIX_CLUSTER']='1';os.environ['MEMOSCHEMA_FIX_ANCHOR_SCHEMA']='1'
    # Explicitly fresh construction; isolate inherited cache settings.
    for k in ['MEMOSCHEMA_PAPER_CACHE_DIRS','MEMOSCHEMA_MERGE_CACHE_DIR','MEMOSCHEMA_SEED_CACHE_DIR']:
        os.environ.pop(k,None)
    out.mkdir(parents=True,exist_ok=True)
    os.environ['MEMOSCHEMA_USAGE_LOG']=str(out/'usage.jsonl')
    sys.path[:0]=[str(ROOT/'Phase1'),str(ROOT/'scripts'),str(ROOT)]
    import schema_induction as si
    from schema_lib.schema_cluster_merge import build_canonical_candidates,select_final_schema
    from schema_lib.subdomain_discovery import discover_subdomain_anchors
    import memexp26 as m
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    papers=si.collect_paper_texts_from_paths(pdfs,max_chars_per_paper=12000)
    if len(papers)!=len(pdfs):raise RuntimeError('Some PDFs could not be read; refusing a silently reduced corpus.')
    rows=si.propose_paper_level_candidate_fields(papers=papers,domain='chem',last_tier=TIER[a.subdomain],max_fields_per_paper=25,max_text_chars=12000,distill_input_chars=60000,distill_quote_limit=10,disable_distill=False)
    pp=out/'paper_level_field_proposals.jsonl';si.save_jsonl(pp,rows)
    canon,_=build_canonical_candidates(paper_level_jsonl=pp,out_dir=out,domain='chem',embed_threshold=0.65)
    for sample in range(1,a.anchor_samples+1):
        sd=out/f's{sample}';sd.mkdir()
        anchors=discover_subdomain_anchors(llm_chat_fn=lambda msgs,force_json=False:si.llm_chat(msgs,force_json=True),paper_level_jsonl=pp,domain='chem',last_tier=TIER[a.subdomain],n_anchors=8,out_path=sd/'anchors.json')
        draft=select_final_schema(canonical_candidates=canon,out_dir=sd,domain='chem',last_tier=TIER[a.subdomain],max_final_fields=8,min_support=2,anchors=anchors)
        si.save_json(sd/'draft_schema.json',draft)
        m.OUT=sd
        fields=m.merge_fields(f'merge_s{sample}',draft,canon,a.subdomain)
        if len(fields)!=8:raise RuntimeError(f'Expected 8 fields, received {len(fields)}; inspect saved call.')
        si.save_json(sd/'final_schema.json',fields)
        for rep in range(1,a.judge_reps+1):
            for label,script in [('hierarchical','evaluate_fine.py'),('strict','evaluate_against_ontology.py')]:
                subprocess.run([sys.executable,str(ROOT/'Evaluation'/script),'--schema',str(sd/'final_schema.json'),'--subdomain',a.subdomain,'--out-dir',str(sd/'evaluation'/f'rep{rep}'/label)],check=True,cwd=ROOT)
if __name__=='__main__':main()
