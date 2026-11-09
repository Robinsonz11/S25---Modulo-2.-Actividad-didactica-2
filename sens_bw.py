"""Sensibilidad de Baum-Welch a n_iter y reinicios (escenario E1_base, mismas secuencias que hmm_red.py, semilla 42)."""
import numpy as np, pandas as pd, sys
from multiprocessing import Pool
import hmm_red as H
CONF=[(100,2),(500,2),(500,8)]
def work(args):
    r,ss=args
    rng=np.random.default_rng(ss); sc=H.SCENARIOS[0]
    A,B=H.make_A(sc["d"]),H.make_B(sc["lam"]); s,o=H.simulate(A,B,H.PI,sc["T"],rng)
    out=[]
    for ni,rs in CONF:
        rg=np.random.default_rng([42,r,ni,rs])
        Ae,Be,pie,ll,it=H.baum_welch(o,rg,restarts=rs,n_iter=ni)
        p=H.best_permutation(Ae,Be,A,B)
        acc=(H.viterbi(Ae[np.ix_(p,p)],Be[p],pie[p],o)==s).mean()
        out.append(dict(n_iter=ni,restarts=rs,rep=r,acc_bw=acc,A_err=np.linalg.norm(Ae[np.ix_(p,p)]-A),B_err=np.linalg.norm(Be[p]-B),iters=it,capped=it>=ni))
    return out
if __name__=="__main__":
    reps=int(sys.argv[1]); ss=np.random.SeedSequence(42).spawn(len(H.SCENARIOS))[0].spawn(reps)
    with Pool() as P: res=P.map(work,list(enumerate(ss)))
    df=pd.DataFrame([x for r in res for x in r]); df.to_csv("sens_bw_raw.csv",index=False)
    print(df.groupby(["n_iter","restarts"]).agg(acc=("acc_bw","mean"),sd=("acc_bw","std"),mn=("acc_bw","min"),A=("A_err","mean"),B=("B_err","mean"),cap=("capped","mean")).round(3))
