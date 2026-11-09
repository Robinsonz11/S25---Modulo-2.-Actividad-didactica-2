# HMM para detección de estados operativos de una red (simulación con verdad conocida)

Estados: Normal, Congestión, Ataque. Observaciones: tráfico bajo, medio, alto, ráfaga anómala.
Implementa desde cero (numpy) simulación, Forward, Backward, Viterbi y Baum-Welch con reinicios aleatorios.

## Instalación
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Reproducir los resultados del artículo
    python hmm_red.py --reps 50 --seed 42 --outdir results      # 6 escenarios x 50 réplicas
    python pruebas_estadisticas.py results/results_raw.csv      # Tabla 8
    python sens_bw.py 30                                        # Tabla 7 (≈ 4 min, 1 núcleo)
    python validar_hmmlearn.py                                  # contraste con hmmlearn
    python hmm_red.py --quick                                   # prueba rápida (5 réplicas)

## Salidas
- results/results_raw.csv: una fila por (escenario, réplica)
- results/summary.csv: media, desv., mín., máx. e IC 95 % por escenario y métrica
- results/fig1_exactitud.png, results/fig2_error_parametros.png (300 dpi)
- results/entorno.json: versiones de Python y librerías, semilla y réplicas
- pruebas_estadisticas.csv, sens_bw_raw.csv

## Reproducibilidad
Semilla maestra 42 (numpy SeedSequence); de ella se derivan las semillas de cada escenario y réplica.
Las versiones usadas se guardan en results/entorno.json; requirements.txt debe coincidir con ese archivo.

## Parámetros del modelo
Definidos al inicio de hmm_red.py (PI, A_BASE, B_BASE, ESCENARIOS). Los valores son ilustrativos, no provienen de tráfico real.

