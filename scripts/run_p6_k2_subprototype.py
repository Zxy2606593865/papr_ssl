#!/usr/bin/env python
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WavLMModel
from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset, frame_collate
from papr_ssl.training.teacher.real_experiment import build_mdsc_audio_catalog, load_real_task_rows, qualified_utt_relpath, resolve_task_audio_path

MODEL_ID="microsoft/wavlm-large"
REV="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
LAYER=15; SR=16000
G={"macro_f1_min":.80,"correct_accept_min":.80,"wrong_intent_max":.10,"known_reject_max":.20,"unknown_reject_min":.85}

def readj(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def loaddr(method,ckpt,device):
    o=torch.load(ckpt,map_location="cpu",weights_only=False); m=build_dr(method); m.load_state_dict(o["dr_state_dict"]); m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)
    return m
@torch.inference_mode()
def embedds(ds,drs,device):
    out={m:{} for m in drs}; dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,collate_fn=frame_collate)
    for b in dl:
        x=b["features"].to(device); fm=b["frame_mask"].to(device)
        for m,dr in drs.items():
            z=dr(x,fm).cpu()
            for u,e in zip(b["utt_ids"],z): out[m][u]=e
    return out
def wav(p):
    d,sr=sf.read(p,dtype="float32",always_2d=True)
    if int(sr)!=SR: raise RuntimeError("sample rate")
    return torch.from_numpy(d).float().mean(1)
@torch.inference_mode()
def embedunk(open_index,root,wanted,backbone,drs,device):
    rows=[r for r in load_real_task_rows(open_index,splits=("dev",)) if r.utt_id in wanted]
    legacy=[r for r in rows if r.audio_relpath is None and qualified_utt_relpath(r) is None]
    cat=build_mdsc_audio_catalog(root) if legacy else None
    out={m:{} for m in drs}
    for i,r in enumerate(rows,1):
        p=resolve_task_audio_path(row=r,mdsc_root=root,audio_catalog=cat)
        x=wav(p).unsqueeze(0).to(device); am=torch.ones_like(x,dtype=torch.long)
        hs=backbone(input_values=x,attention_mask=am,output_hidden_states=True,return_dict=True).hidden_states[LAYER].float()
        fm=backbone._get_feature_vector_attention_mask(hs.shape[1],am).bool()
        for m,dr in drs.items(): out[m][r.utt_id]=dr(hs,fm).squeeze(0).cpu()
        if i%100==0 or i==len(rows): print(f"[unknown dev] {i}/{len(rows)}")
    return out
def subprotos(enroll,emb):
    p=torch.empty((30,2,64))
    for r in enroll: p[int(r["label_index"]),int(r["subprototype_index"])]=F.normalize(emb[r["utt_id"]],p=2,dim=0)
    return p
def scores(rows,emb,subp,known):
    z=F.normalize(torch.stack([emb[r["utt_id"]] for r in rows]),p=2,dim=-1)
    cs=torch.einsum("bd,ckd->bck",z,subp).max(-1).values
    t=torch.topk(cs,2,dim=1)
    o={"score":t.values[:,0].numpy(),"margin":(t.values[:,0]-t.values[:,1]).numpy(),"pred":t.indices[:,0].numpy()}
    if known:o["true"]=np.asarray([int(r["label_index"]) for r in rows])
    return o
def mf1(y,p,a):
    fs=[]
    for c in range(30):
        pc=a&(p==c); tc=y==c; tp=int(np.sum(pc&tc)); fp=int(np.sum(pc&~tc)); fn=int(np.sum(~pc&tc))
        pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; fs.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return float(np.mean(fs))
def met(k,u,ts,tm):
    ka=(k["score"]>=ts)&(k["margin"]>=tm); ua=(u["score"]>=ts)&(u["margin"]>=tm)
    return {"macro_f1":mf1(k["true"],k["pred"],ka),"correct_accept":float(np.mean(ka&(k["pred"]==k["true"]))),
            "wrong_intent":float(np.mean(ka&(k["pred"]!=k["true"]))),"known_reject":float(np.mean(~ka)),
            "unknown_reject":float(np.mean(~ua)),"unknown_accept_far":float(np.mean(ua))}
def ok(m): return m["macro_f1"]>=.8 and m["correct_accept"]>=.8 and m["wrong_intent"]<=.1 and m["known_reject"]<=.2 and m["unknown_reject"]>=.85
def viol(m):
    return max(0,.8-m["macro_f1"])/.8+max(0,.8-m["correct_accept"])/.8+max(0,m["wrong_intent"]-.1)/.1+max(0,m["known_reject"]-.2)/.2+max(0,.85-m["unknown_reject"])/.85
def grid(v):
    q=np.unique(np.quantile(v,np.linspace(0,1,101))); return [float(q[0]-1e-6)]+[float(x) for x in q]+[float(q[-1]+1e-6)]
def calibrate(k,u):
    feas=[]; diag=[]
    for ts in grid(np.r_[k["score"],u["score"]]):
        for tm in grid(np.r_[k["margin"],u["margin"]]):
            m=met(k,u,ts,tm); r={"score_threshold":ts,"margin_threshold":tm,"metrics":m}
            if ok(m):feas.append(r)
            else:r["violation"]=viol(m);diag.append(r)
    if feas:
        b=max(feas,key=lambda x:(x["metrics"]["macro_f1"],x["metrics"]["correct_accept"],x["metrics"]["unknown_reject"]));b["calibration_feasible"]=True;return b
    b=min(diag,key=lambda x:(x["violation"],-x["metrics"]["macro_f1"]));b["calibration_feasible"]=False;return b
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protocol-dir",type=Path,default=Path("artifacts/p6_dev_k2/protocol"))
    ap.add_argument("--frame-cache",type=Path,default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"))
    ap.add_argument("--core-index",type=Path,default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"))
    ap.add_argument("--open-index",type=Path,default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30_open_set_eval.index.jsonl"))
    ap.add_argument("--mdsc-root",type=Path,default=Path("datasets/public/mdsc"))
    ap.add_argument("--output-dir",type=Path,default=Path("artifacts/p6_dev_k2/qualification"))
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pr=json.loads((a.protocol_dir/"p6_k2_protocol.json").read_text(encoding="utf-8"))
    enroll=readj(a.protocol_dir/"generic_enrollment_k2_subproto.jsonl");kc=readj(a.protocol_dir/"generic_dev_cal_known.jsonl");ks=readj(a.protocol_dir/"generic_dev_score_known.jsonl");uc=readj(a.protocol_dir/"generic_dev_cal_unknown.jsonl");us=readj(a.protocol_dir/"generic_dev_score_unknown.jsonl")
    device=torch.device(a.device);drs={};meta={}
    for m in ("mean","attention"):
        md=pr["representative_checkpoints"][m];cp=Path(md["checkpoint"])
        if sha(cp)!=md["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
        drs[m]=loaddr(m,cp,device);meta[m]=md
    tr=FrameCacheDataset(cache_dir=a.frame_cache,task_index=a.core_index,split="train")
    dv=FrameCacheDataset(cache_dir=a.frame_cache,task_index=a.core_index,split="dev",class_to_index=tr.class_to_index)
    te=embedds(tr,drs,device);de=embedds(dv,drs,device)
    wanted={r["utt_id"] for r in uc+us}
    bb=WavLMModel.from_pretrained(MODEL_ID,revision=REV,output_hidden_states=True,local_files_only=True).eval().to(device)
    for p in bb.parameters():p.requires_grad_(False)
    ue=embedunk(a.open_index,a.mdsc_root,wanted,bb,drs,device)
    results={}
    for m in ("mean","attention"):
        sp=subprotos(enroll,te[m]); ck=scores(kc,de[m],sp,True); cu=scores(uc,ue[m],sp,False);cal=calibrate(ck,cu)
        sk=scores(ks,de[m],sp,True);su=scores(us,ue[m],sp,False);mm=met(sk,su,cal["score_threshold"],cal["margin_threshold"])
        results[m]={"checkpoint":meta[m],"calibration":cal,"generic_dev_score_metrics":mm,"generic_dev_score_gate_pass":bool(cal["calibration_feasible"] and ok(mm))}
        print("="*100);print(f"P6 K2 {m.upper()}");print("calibration_feasible =",cal["calibration_feasible"]);print("thresholds =",cal["score_threshold"],cal["margin_threshold"]);print(json.dumps(mm,indent=2))
    drop=results["mean"]["generic_dev_score_metrics"]["macro_f1"]-results["attention"]["generic_dev_score_metrics"]["macro_f1"]
    e0b=drop<=.02;eligible=results["attention"]["generic_dev_score_gate_pass"] and e0b
    out={"schema":"papr_ssl.p6_k2_subprototype_qualification.v1","results":results,
         "p6_04_mainline_absolute_gate_pass":results["attention"]["generic_dev_score_gate_pass"],
         "p6_05_e0b":{"baseline_minus_mainline_drop":drop,"max_allowed_drop":.02,"pass":e0b},
         "eligible_for_p6_06_freeze":eligible,"generic_test":"sealed_not_accessed"}
    (a.output_dir/"p6_k2_dev_gate.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("="*100);print("P6 K2 DEVELOPMENT GATE SUMMARY");print("P6-04 absolute Gate:", "PASS" if out["p6_04_mainline_absolute_gate_pass"] else "FAIL");print("P6-05 E0b Gate:", "PASS" if e0b else "FAIL");print("eligible for P6-06 freeze:",eligible);print("generic_test accessed: NO")
if __name__=="__main__": main()
