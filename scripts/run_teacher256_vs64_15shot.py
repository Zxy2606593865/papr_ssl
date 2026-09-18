#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def query_features(query, enrollment):
    q = l2norm(query)
    e = l2norm(enrollment)
    centroids = l2norm(np.mean(e, axis=1))
    scores = q @ centroids.T
    order = np.argsort(scores, axis=1)
    pred = order[:, -1]
    second = order[:, -2]
    row = np.arange(len(q))
    s1 = scores[row, pred]
    s2 = scores[row, second]
    margin = s1 - s2
    pe = e[pred]
    sims = np.einsum("nd,nkd->nk", q, pe)
    ss = np.sort(sims, axis=1)
    mean15 = np.mean(sims, axis=1)
    std15 = np.std(sims, axis=1)
    min15 = ss[:,0]
    q25 = np.quantile(sims, .25, axis=1)
    med = np.median(sims, axis=1)
    q75 = np.quantile(sims, .75, axis=1)
    max15 = ss[:,-1]
    top3 = np.mean(ss[:,-3:], axis=1)
    top5 = np.mean(ss[:,-5:], axis=1)
    bottom3 = np.mean(ss[:,:3], axis=1)
    x = np.stack(
        [s1, margin, mean15, std15, min15, q25, med, q75, max15,
         top3, top5, bottom3, s1-mean15, max15-min15],
        axis=1
    )
    return pred.astype(np.int64), x.astype(np.float64)


def split_known(y, seed):
    rng = random.Random(seed)
    cal, score = [], []
    for c in range(30):
        idx = np.flatnonzero(y == c).tolist()
        rng.shuffle(idx)
        n = len(idx)//2
        cal.extend(idx[:n]); score.extend(idx[n:])
    return np.asarray(sorted(cal)), np.asarray(sorted(score))


def split_unknown(n, seed):
    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    k = n//2
    return np.asarray(sorted(idx[:k])), np.asarray(sorted(idx[k:]))


def standardizer(x):
    m = x.mean(0)
    s = x.std(0)
    return m, np.where(s < 1e-12, 1.0, s)


def zscore(x, m, s):
    return (x-m)/s


def fit_logistic(x, y, l2=1e-2):
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    n_pos = float((y==1).sum()); n_neg = float((y==0).sum()); n=float(len(y))
    wp=n/(2*n_pos); wn=n/(2*n_neg)
    sw = torch.where(yt>0.5, torch.full_like(yt,wp), torch.full_like(yt,wn))
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w,b], lr=1.0, max_iter=300,
                            tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        logits = xt@w+b
        loss = (sw*torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt, reduction="none")).mean() + l2*(w*w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def sigmoid(x,w,b):
    a=x@w+b
    out=np.empty_like(a)
    pos=a>=0
    out[pos]=1/(1+np.exp(-a[pos]))
    e=np.exp(a[~pos]); out[~pos]=e/(1+e)
    return out


def macro_f1(acc,true,pred):
    f=[]
    for c in range(30):
        pc=acc&(pred==c); tc=true==c
        tp=int((pc&tc).sum()); fp=int((pc&~tc).sum()); fn=int((~pc&tc).sum())
        den=2*tp+fp+fn
        f.append((2*tp/den) if den else 0.0)
    return float(np.mean(f))


def metrics(true,pred,kp,up,tau):
    ka=kp>=tau; ua=up>=tau
    correct=ka&(pred==true); wrong=ka&(pred!=true)
    return {
        "macro_f1": macro_f1(ka,true,pred),
        "correct_accept": float(correct.mean()),
        "wrong_intent": float(wrong.mean()),
        "known_reject": float((~ka).mean()),
        "unknown_reject": float((~ua).mean()),
        "unknown_accept_far": float(ua.mean()),
    }


def gate(m):
    return (
        m["macro_f1"]>=GATES["macro_f1_min"] and
        m["correct_accept"]>=GATES["correct_accept_min"] and
        m["wrong_intent"]<=GATES["wrong_intent_max"] and
        m["known_reject"]<=GATES["known_reject_max"] and
        m["unknown_reject"]>=GATES["unknown_reject_min"]
    )


def viol(m):
    return (
        max(0,GATES["macro_f1_min"]-m["macro_f1"])/GATES["macro_f1_min"] +
        max(0,GATES["correct_accept_min"]-m["correct_accept"])/GATES["correct_accept_min"] +
        max(0,m["wrong_intent"]-GATES["wrong_intent_max"])/GATES["wrong_intent_max"] +
        max(0,m["known_reject"]-GATES["known_reject_max"])/GATES["known_reject_max"] +
        max(0,GATES["unknown_reject_min"]-m["unknown_reject"])/GATES["unknown_reject_min"]
    )


def calibrate(true,pred,kp,up):
    u=np.unique(np.concatenate([kp,up]))
    cand=np.concatenate([[np.nextafter(u[0],-np.inf)],u,[np.nextafter(u[-1],np.inf)]])
    bestf=None; bestfk=None; bestd=None; bestdk=None; feasible=0
    for tau in cand:
        m=metrics(true,pred,kp,up,float(tau))
        if gate(m):
            feasible+=1
            key=(m["macro_f1"],m["correct_accept"],m["unknown_reject"],-m["wrong_intent"],-m["known_reject"],tau)
            if bestfk is None or key>bestfk:
                bestfk=key; bestf=(float(tau),m)
        else:
            v=viol(m); key=(-v,m["macro_f1"],m["correct_accept"],m["unknown_reject"])
            if bestdk is None or key>bestdk:
                bestdk=key; bestd=(float(tau),m,v)
    if bestf:
        return True,bestf[0],feasible
    return False,bestd[0],0


def summary(vals):
    x=np.asarray(vals,dtype=np.float64)
    return {"mean":float(x.mean()),"sample_std":float(x.std(ddof=1)),
            "median":float(np.median(x)),"min":float(x.min()),"max":float(x.max())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--embeddings",type=Path,
                    default=Path("artifacts/p6_teacher_256_15shot/embeddings/teacher_256d_15shot_embeddings.npz"))
    ap.add_argument("--reference64",type=Path,
                    default=Path("artifacts/p6_15shot_consistency/repeated_splits/p6_15shot_consistency_results.json"))
    ap.add_argument("--split-policy",type=Path,
                    default=Path("artifacts/p6_15shot_stability/repeated_splits/p6_15shot_stability.json"))
    ap.add_argument("--output-dir",type=Path,
                    default=Path("artifacts/p6_teacher_256_15shot/repeated_splits"))
    args=ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref=json.loads(args.reference64.read_text(encoding="utf-8"))
    splitp=json.loads(args.split_policy.read_text(encoding="utf-8"))
    if ref["generic_test"]!="sealed_not_accessed" or splitp["generic_test"]!="sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    d=np.load(args.embeddings,allow_pickle=False)
    enrollment=d["enrollment"].astype(np.float64)
    ke=d["known_embedding"].astype(np.float64)
    ky=d["known_true"].astype(np.int64)
    ue=d["unknown_embedding"].astype(np.float64)
    if enrollment.shape!=(30,15,256) or ke.shape!=(442,256) or ue.shape!=(1178,256):
        raise RuntimeError("unexpected 256D embedding shapes")

    kp_cls,kx=query_features(ke,enrollment)
    _,ux=query_features(ue,enrollment)

    n=int(splitp["num_splits"]); base=int(splitp["base_seed"])
    runs=[]
    for i in range(n):
        seed=base+i
        kc,ks=split_known(ky,seed)
        uc,us=split_unknown(len(ue),seed+1000003)
        xcal=np.concatenate([kx[kc],ux[uc]],0)
        ycal=np.concatenate([np.ones(len(kc),dtype=np.int64),np.zeros(len(uc),dtype=np.int64)])
        m,s=standardizer(xcal)
        w,b=fit_logistic(zscore(xcal,m,s),ycal)
        pkc=sigmoid(zscore(kx[kc],m,s),w,b)
        puc=sigmoid(zscore(ux[uc],m,s),w,b)
        feasible,tau,_=calibrate(ky[kc],kp_cls[kc],pkc,puc)

        pks=sigmoid(zscore(kx[ks],m,s),w,b)
        pus=sigmoid(zscore(ux[us],m,s),w,b)
        met=metrics(ky[ks],kp_cls[ks],pks,pus,tau)
        passed=bool(feasible and gate(met))

        r64=ref["runs"][i]
        if int(r64["seed"])!=seed:
            raise RuntimeError("64D reference split seed mismatch")
        m64=r64["score_metrics"]
        delta={name:float(met[name]-m64[name]) for name in
               ("macro_f1","correct_accept","wrong_intent","known_reject","unknown_reject","unknown_accept_far")}
        runs.append({
            "split_index":i,"seed":seed,"calibration_feasible":feasible,
            "probability_threshold":float(tau),"score_metrics":met,
            "absolute_gate_pass":passed,"delta_vs_64d_consistency":delta
        })
        print(
            f"[split {i+1:02d}/{n}] cal_feasible={feasible} gate={'PASS' if passed else 'FAIL'} "
            f"F1={met['macro_f1']:.4f} CA={met['correct_accept']:.4f} "
            f"WI={met['wrong_intent']:.4f} KR={met['known_reject']:.4f} "
            f"UR={met['unknown_reject']:.4f} | "
            f"dF1={delta['macro_f1']:+.4f} dCA={delta['correct_accept']:+.4f} dUR={delta['unknown_reject']:+.4f}"
        )

    names=("macro_f1","correct_accept","wrong_intent","known_reject","unknown_reject","unknown_accept_far")
    sm={k:summary([r["score_metrics"][k] for r in runs]) for k in names}
    dm={k:summary([r["delta_vs_64d_consistency"][k] for r in runs]) for k in names}

    out={
        "schema":"papr_ssl.p6_teacher_256_vs_64_15shot.v1",
        "embedding_dim":256,
        "num_splits":n,
        "calibration_feasible_rate":float(np.mean([r["calibration_feasible"] for r in runs])),
        "absolute_gate_pass_count":int(sum(r["absolute_gate_pass"] for r in runs)),
        "absolute_gate_pass_rate":float(np.mean([r["absolute_gate_pass"] for r in runs])),
        "score_metric_summary":sm,
        "delta_vs_64d_consistency_summary":dm,
        "reference_64d_summary":ref["score_metric_summary"],
        "runs":runs,
        "project_2_canonical_64d":"preserved_not_modified",
        "generic_test":"sealed_not_accessed",
    }
    (args.output_dir/"p6_teacher_256_vs_64_15shot.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("="*108)
    print("P6 256D vs 64D — 15-SHOT PERSONALIZED OPEN-SET SUMMARY")
    print(f"calibration feasible: {sum(r['calibration_feasible'] for r in runs)}/{n} ({out['calibration_feasible_rate']:.3f})")
    print(f"absolute Gate PASS:   {out['absolute_gate_pass_count']}/{n} ({out['absolute_gate_pass_rate']:.3f})")
    print("-"*108)
    for k in ("macro_f1","correct_accept","wrong_intent","known_reject","unknown_reject"):
        print(f"{k:20s} 256D={sm[k]['mean']:.6f} delta_vs_64D={dm[k]['mean']:+.6f}")
    print("-"*108)
    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    print("P6 256D vs 64D STATUS: COMPLETE")


if __name__=="__main__":
    main()
