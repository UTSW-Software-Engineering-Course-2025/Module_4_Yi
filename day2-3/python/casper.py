# casper.py
# ---------------------------------------------
# Python port of pipeline_CaSpEr_SWE_template.m
# ---------------------------------------------
import time, pathlib, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils  import jordan_decomp, plot_activity_map, plot_binary_map
from arhmm  import em_arhmm, viterbi_path, bic_k, one_state_bic

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------  参数  -----------------
BASE         = pathlib.Path("D:/document/UTSW/software_en/Module_4_Yi/day2-3")
RAW_DIR      = BASE  / "raw"
OUT_DIR      = BASE  / "output_py" / "HMM_py"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_NAME     = "actSig_HCLindexed.csv"
NUM_STATES   = 5                               # K = 5  (包含降解状态)
EM_TOL       = 1e-9
MAX_ITER     = 5000

# ------------  读入  -----------------
actmap       = pd.read_csv(RAW_DIR / CSV_NAME, header=None).to_numpy(dtype=float)
actmap = actmap[0:1,:]
num_neuron, num_frames = actmap.shape

# 可视化原始热图 (与 MATLAB 相同)
plot_activity_map(actmap, OUT_DIR / "actmap.png")

# ------------  一次性数组 (结果容器) -------------
hmm_states   = np.zeros_like(actmap, dtype=int)
hmm_binary   = np.zeros_like(actmap, dtype=int)
hmm_logL     = np.zeros(num_neuron)
hmm_bic      = np.zeros(num_neuron)

coef_phi0    = np.zeros((num_neuron, NUM_STATES-1))
coef_phi1    = np.zeros_like(coef_phi0)
coef_sigma   = np.zeros_like(coef_phi0)

one_state_b  = np.zeros(num_neuron)

# ------------  Step-0: 单状态检验 --------------
print(">> One-state (no-spike) baseline ...")
for nid in range(num_neuron):
    ca   = actmap[nid]
    yP = jordan_decomp(ca)             # 返回正增序列
    one_state_b[nid] = one_state_bic(yP)
print("   done.\n")

# ------------  Step-1: 逐神经元 EM --------------
print(">> EM training (AR-HMM) ...")
tic = time.time()

for nid in range(num_neuron):
    print(f"Processing neuron {nid+1} of {num_neuron}")
    ca            = actmap[nid]
    # -- Jordan decomposition --
    yP, mask_P    = jordan_decomp(ca, return_mask=True)
    yP1           = np.roll(yP, 1);  yP1[0] = 0.0

    # -- EM --
    res           = em_arhmm(yP, yP1,
                             M            = NUM_STATES-1,
                             max_iter     = MAX_ITER,
                             tol          = EM_TOL)
    states_P, logL, params = res

    # -- Viterbi for hard labels (可选：直接用 γ 最大) --
    fitted_P   = viterbi_path(yP, yP1, params)

    # -- 把排除的递减期补回 --
    fitted_all         = np.zeros_like(ca, dtype=int)
    fitted_all[mask_P] = fitted_P + 1     # +1 保证 1=递减态
    hmm_states[nid]    = fitted_all
    hmm_binary[nid]    = (fitted_all == (NUM_STATES-1)).astype(int)

    # -- 记录统计量 --
    hmm_logL[nid]      = logL
    hmm_bic[nid]       = -2*logL + bic_k(NUM_STATES-1)*np.log(len(yP))

    φ0, φ1, σ2         = params["phi0"], params["phi1"], params["sigmasq"]
    idx_sort           = np.argsort(φ0 + φ1)                 # 按均值排序重标号
    coef_phi0[nid]     = φ0[idx_sort]
    coef_phi1[nid]     = φ1[idx_sort]
    coef_sigma[nid]    = σ2[idx_sort]

tsec = time.time()-tic
print(f"   done in {tsec/60:.1f} min.\n")

# ------------  保存 CSV / JSON --------------
np.savetxt(OUT_DIR / "hmm_statemap.csv",   hmm_states, fmt='%d', delimiter=',')
np.savetxt(OUT_DIR / "hmm_binarymap.csv",  hmm_binary, fmt='%d', delimiter=',')
np.savetxt(OUT_DIR / "hmm_logL.csv",       hmm_logL,   delimiter=',')
np.savetxt(OUT_DIR / "arhmm_coef_phi0.csv",coef_phi0,  delimiter=',')
np.savetxt(OUT_DIR / "arhmm_coef_phi1.csv",coef_phi1,  delimiter=',')
np.savetxt(OUT_DIR / "arhmm_coef_sigma.csv",coef_sigma,delimiter=',')
plot_binary_map(hmm_binary, OUT_DIR / "hmm_binarymap.png")

print(">> Multi-state model better than one-state for ALL neurons?",
      np.all(one_state_b > hmm_bic))
