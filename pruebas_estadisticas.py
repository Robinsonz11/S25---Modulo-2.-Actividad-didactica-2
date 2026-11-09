"""Pruebas de Welch, Mann-Whitney y d de Cohen sobre results_raw.csv (comparaciones entre escenarios)."""
import sys, numpy as np, pandas as pd
from scipy import stats
raw = sys.argv[1] if len(sys.argv) > 1 else "results/results_raw.csv"
d = pd.read_csv(raw)
COMP = [("E2_volatil","E1_base","acc_viterbi"),("E2_persist","E1_base","acc_viterbi"),
        ("E3_ruido","E1_base","acc_viterbi"),("E3_ruido","E1_base","f1_ataque_viterbi"),
        ("E4_T100","E4_T2000","acc_viterbi"),("E4_T100","E4_T2000","A_err"),
        ("E4_T100","E4_T2000","B_err"),("E1_base","E4_T2000","A_err")]
rows=[]
for a,b,m in COMP:
    x=d[d.escenario==a][m].dropna(); y=d[d.escenario==b][m].dropna()
    sp=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    rows.append(dict(A=a,B=b,metrica=m,dif_medias=x.mean()-y.mean(),
        p_welch=stats.ttest_ind(x,y,equal_var=False).pvalue,
        p_mannwhitney=stats.mannwhitneyu(x,y).pvalue,d_cohen=(x.mean()-y.mean())/sp))
out=pd.DataFrame(rows); out["signif_bonferroni"]=out.p_welch<0.05/len(out)
out.to_csv("pruebas_estadisticas.csv",index=False); print(out.to_string(index=False))
print("\nViterbi verdadero vs Baum-Welch (t pareada por escenario):")
for s,g in d.groupby("escenario"):
    print(s,"dif=%.3f"%(g.acc_viterbi-g.acc_bw).mean(),"p=%.2g"%stats.ttest_rel(g.acc_viterbi,g.acc_bw).pvalue)
