"""Contraste de la implementación propia (hmm_red.py) con hmmlearn."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import hmm_red as H
from hmmlearn.hmm import CategoricalHMM
rng=np.random.default_rng(42)
maxdll=0;agree=[];accs_h=[];accs_o=[];lls_h=[];lls_o=[]
for r in range(30):
    s,o=H.simulate(H.A_BASE,H.B_BASE,H.PI,500,rng)
    m=CategoricalHMM(n_components=3,n_features=4); m.startprob_=H.PI;m.transmat_=H.A_BASE;m.emissionprob_=H.B_BASE
    X=o.reshape(-1,1)
    ll_h=m.score(X); _,c=H.forward(H.A_BASE,H.B_BASE,H.PI,o); ll_o=np.log(c).sum()
    maxdll=max(maxdll,abs(ll_h-ll_o))
    ph=m.predict(X); po=H.viterbi(H.A_BASE,H.B_BASE,H.PI,o); agree.append((ph==po).mean())
print("max |dLogL| forward:",maxdll); print("coincidencia Viterbi:",np.mean(agree))
# Baum-Welch: hmmlearn con 2 reinicios aleatorios vs propio, mismas secuencias
res=[]
for r in range(30):
    s,o=H.simulate(H.A_BASE,H.B_BASE,H.PI,500,rng); X=o.reshape(-1,1)
    best=None
    for k in range(2):
        m=CategoricalHMM(n_components=3,n_iter=100,tol=1e-6*500,init_params="ste",random_state=int(rng.integers(1e9)))
        m.fit(X)
        if best is None or m.score(X)>best.score(X): best=m
    p=H.best_permutation(best.transmat_,best.emissionprob_,H.A_BASE,H.B_BASE)
    A_=best.transmat_[np.ix_(p,p)];B_=best.emissionprob_[p];pi_=best.startprob_[p]
    acc_h=(H.viterbi(A_,B_,pi_,o)==s).mean()
    Ae,Be,pie,lle,it=H.baum_welch(o,rng); pp=H.best_permutation(Ae,Be,H.A_BASE,H.B_BASE)
    acc_o=(H.viterbi(Ae[np.ix_(pp,pp)],Be[pp],pie[pp],o)==s).mean()
    res.append((acc_h,acc_o,np.linalg.norm(A_-H.A_BASE),np.linalg.norm(Ae[np.ix_(pp,pp)]-H.A_BASE)))
res=np.array(res);print("acc BW hmmlearn %.3f±%.3f | propio %.3f±%.3f"%(res[:,0].mean(),res[:,0].std(ddof=1),res[:,1].mean(),res[:,1].std(ddof=1)))
print("A_err hmmlearn %.3f | propio %.3f"%(res[:,2].mean(),res[:,3].mean()))
