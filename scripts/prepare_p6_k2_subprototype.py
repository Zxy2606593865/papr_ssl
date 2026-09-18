#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset

SEEDS=(17,29,43)
METHODS=("mean","attention")
SALT="papr_ssl_p6_protocol_v1"

def key(*parts):
    h=hashlib.sha256(SALT.encode())
    for p in parts:
        h.update(b"\\0"); h.update(str(p).encode())
    return h.hexdigest()

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(p, rows):
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def representative(root, method):
    rows=[]
    for seed in SEEDS:
        o=json.loads((root/method/f"seed_{seed:04d}"/"p5_result.json").read_text(encoding="utf-8"))
        rows.append({"seed":seed,"score":float(o["selected_generic_dev_score"]),
                     "selected_epoch":int(o["selected_epoch"]),
                     "checkpoint":o["selected_checkpoint"],
                     "checkpoint_sha256":o["selected_checkpoint_sha256"]})
    return sorted(rows,key=lambda x:(x["score"],x["seed"]))[1]

def groups(ds):
    g={}
    for r in ds.rows: g.setdefault(r["label_text"],[]).append(r["utt_id"])
    return g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frame-cache",type=Path,default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"))
    ap.add_argument("--core-index",type=Path,default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"))
    ap.add_argument("--open-index",type=Path,default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30_open_set_eval.index.jsonl"))
    ap.add_argument("--p5-run-root",type=Path,default=Path("artifacts/p5_runs/mean_vs_attention"))
    ap.add_argument("--output-dir",type=Path,default=Path("artifacts/p6_dev_k2/protocol"))
    a=ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    tr=FrameCacheDataset(cache_dir=a.frame_cache,task_index=a.core_index,split="train")
    dv=FrameCacheDataset(cache_dir=a.frame_cache,task_index=a.core_index,split="dev",class_to_index=tr.class_to_index)
    tg,dg=groups(tr),groups(dv)
    enroll=[]; kc=[]; ks=[]
    for label in sorted(tg):
        tids=sorted(tg[label],key=lambda u:key("enrollment",label,u))[:2]
        for j,u in enumerate(tids):
            enroll.append({"utt_id":u,"label_text":label,"label_index":tr.class_to_index[label],
                           "subprototype_index":j,"source_split":"train"})
        dids=sorted(dg[label],key=lambda u:key("known_dev",label,u))
        n=len(dids)//2
        for u in dids[:n]:
            kc.append({"utt_id":u,"label_text":label,"label_index":tr.class_to_index[label],"source_split":"dev"})
        for u in dids[n:]:
            ks.append({"utt_id":u,"label_text":label,"label_index":tr.class_to_index[label],"source_split":"dev"})
    od=[r for r in read_jsonl(a.open_index) if r.get("split")=="dev"]
    if len(od)!=1178: raise RuntimeError(f"expected 1178 open dev, got {len(od)}")
    od=sorted(od,key=lambda r:key("unknown_dev",r["utt_id"]))
    n=len(od)//2
    uc=[{"utt_id":r["utt_id"],"source_split":"dev"} for r in od[:n]]
    us=[{"utt_id":r["utt_id"],"source_split":"dev"} for r in od[n:]]
    write_jsonl(a.output_dir/"generic_enrollment_k2_subproto.jsonl",enroll)
    write_jsonl(a.output_dir/"generic_dev_cal_known.jsonl",kc)
    write_jsonl(a.output_dir/"generic_dev_score_known.jsonl",ks)
    write_jsonl(a.output_dir/"generic_dev_cal_unknown.jsonl",uc)
    write_jsonl(a.output_dir/"generic_dev_score_unknown.jsonl",us)
    obj={"schema":"papr_ssl.p6_k2_subprototype_protocol.v1","subprototype_k":2,
         "definition":"two independent enrollment embeddings per class; class score=max cosine over the two",
         "representative_checkpoints":{m:representative(a.p5_run_root,m) for m in METHODS},
         "counts":{"enrollment":len(enroll),"known_cal":len(kc),"known_score":len(ks),"unknown_cal":len(uc),"unknown_score":len(us)},
         "generic_test":"sealed_not_accessed"}
    (a.output_dir/"p6_k2_protocol.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("P6 K2 PROTOCOL STATUS: PASS")
    print(obj["counts"]); print("generic_test accessed: NO")
if __name__=="__main__": main()
