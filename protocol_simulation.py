"""
Protocol Simulation — Cryptographic Security Analysis
======================================================
FYP: Simulation and Comparative Analysis of Quantum and Classical
     Cryptographic Protocols Under Quantum Threat Models

Author : Muhammad Noor Akma Bin Abdullah Alinoor (B20230279)
         BSc (Hons) Information Security, Multimedia University Malaysia

Experiments:
  1 — BB84 key generation efficiency vs qubit count
  2 — BB84 QBER analysis with and without eavesdropper
  3 — E91 CHSH parameter analysis with and without eavesdropper
  4 — DH and ECC classical attack complexity vs key/curve size
  5 — RSA key generation time vs key size
  6 — AES-128 vs AES-256 performance and Grover's threat
  7 — SHA-256 hashing performance and Grover's impact
  8 — Cross-protocol comparison (all 8 protocols unified)
  9 — Shor's algorithm (Qiskit QPE + inverse QFT) order-finding on small moduli

Run: python protocol_simulation.py
Outputs: results/  (CSV files)  |  graphs/  (PNG files)

─────────────────────────────────────────────────────────────────
SOURCES AND REFERENCES FOR SIMULATION CODE
─────────────────────────────────────────────────────────────────

[BB84 Protocol Logic]
  Bennett, C. H., & Brassard, G. (2011). Quantum cryptography: Public key
  distribution and coin tossing. Theoretical Computer Science, 560, 7-11.
  (Original work published 1984.)

  Kushwah, A., Akanksha, & Jain, A. (2023). Simulating the BB84 Protocol.
  International Journal for Research in Applied Science & Engineering Technology.

  Quantum Blockchains. (2022). QKD Protocol Simulation with Qiskit.
  https://www.quantumblockchains.io/qkd-protocol-simulation-with-qiskit/

  Adu-Kyere, A., Nigussie, E., & Isoaho, J. (2022). Quantum Key Distribution:
  Modeling and Simulation through BB84 Protocol Using Python3.
  Sensors, 22(16), 6284. https://doi.org/10.3390/s22166284

[E91 Protocol Logic and CHSH Inequality]
  Ekert, A. K. (1991). Quantum cryptography based on Bell's theorem.
  Physical Review Letters, 67(6), 661-663.

  Begimbayeva, Y., & Zhaxalykov, T. (2022). Research of Quantum Key Distribution
  Protocols: BB84, B92, E91. Scientific Journal of Astana IT University.
  https://doi.org/10.37943/QRKJ7456

  Clauser, J. F., Horne, M. A., Shimony, A., & Holt, R. A. (1969).
  Proposed experiment to test local hidden-variable theories.
  Physical Review Letters, 23(15), 880-884.

[Diffie-Hellman Key Exchange and BSGS Attack]
  Diffie, W., & Hellman, M. (1976). New directions in cryptography.
  IEEE Transactions on Information Theory, 22(6), 644-654.

  Shanks, D. (1971). Class number, a theory of factorization, and genera.
  Proceedings of Symposia in Pure Mathematics, 20, 415-440.
  (Original description of Baby-step Giant-step algorithm.)

[Elliptic Curve Cryptography and Brute-force ECDLP]
  Miller, V. S. (1986). Use of elliptic curves in cryptography.
  In H. C. Williams (Ed.), Advances in Cryptology — CRYPTO '85 Proceedings
  (Lecture Notes in Computer Science, Vol. 218, pp. 417-426). Springer.

  Koblitz, N. (1987). Elliptic curve cryptosystems.
  Mathematics of Computation, 48(177), 203-209.

[RSA Algorithm]
  Rivest, R. L., Shamir, A., & Adleman, L. (1978). A method for obtaining digital
  signatures and public-key cryptosystems.
  Communications of the ACM, 21(2), 120-126.

  Python cryptography library (v43.x). (2024). PyCA Cryptography.
  https://cryptography.io/en/latest/

[AES Encryption]
  National Institute of Standards and Technology (NIST). (2001).
  Advanced Encryption Standard (AES). FIPS Publication 197.
  https://doi.org/10.6028/NIST.FIPS.197

  Python cryptography library (v43.x). (2024). PyCA Cryptography.
  https://cryptography.io/en/latest/

[SHA-256 Hash Function]
  National Institute of Standards and Technology (NIST). (2015).
  Secure Hash Standard (SHS). FIPS Publication 180-4.
  https://doi.org/10.6028/NIST.FIPS.180-4

  Python standard library: hashlib module.
  https://docs.python.org/3/library/hashlib.html

[Shor's Algorithm — Order Finding via Quantum Phase Estimation]
  Shor, P. W. (1994). Algorithms for quantum computation: Discrete logarithms
  and factoring. Proceedings of the 35th Annual Symposium on Foundations of
  Computer Science (FOCS), 124-134.

  Shor, P. W. (1997). Polynomial-time algorithms for prime factorization and
  discrete logarithms on a quantum computer. SIAM Journal on Computing, 26(5),
  1484-1509.

  IBM Qiskit Documentation. (2024). Quantum Phase Estimation and QFT.
  https://docs.quantum.ibm.com/

  Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
  Information (10th anniversary ed.). Cambridge University Press.
  (Chapter 5: The quantum Fourier transform and its applications, pp. 216-261.)

[Grover's Algorithm — Analytical Modelling]
  Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
  Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC),
  212-219.

  Mavroeidis, V., Vishi, K., Zych, M., & Josang, A. (2018). The impact of quantum
  computing on present cryptography. International Journal of Advanced Computer
  Science and Applications, 9(3). https://doi.org/10.14569/ijacsa.2018.090354

[Quantum Threat Classification Framework]
  Jebadurai, I. J., & Paulraj, G. J. L. (2025). A security-centric evaluation of
  classical and quantum key exchange protocols for securing the next generation of
  communication. 2025 5th International Conference on Soft Computing for Security
  Applications (ICSCSA), 85-90. IEEE.

  Tripathi, T., Awasthi, A., Singh, S. P., & Chaturvedi, A. (2024). Post quantum
  cryptography and its comparison with classical cryptography.
  arXiv preprint arXiv:2403.19299.

[Simulation Framework and Methodology]
  Pathare, S., Patil, V., & Shah, R. (2025). Experimental analysis on the BB84
  quantum key distribution protocol using Qiskit and IBM Quantum hardware.

  Dhakal, S. (2025). Performance analysis of different quantum key distribution
  protocols for optimised security and efficiency.
  IET Quantum Communication. https://doi.org/10.1049/qtc2.70015
─────────────────────────────────────────────────────────────────
"""

import os, time, math, random, csv, hashlib, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Output directories ────────────────────────────────────────
os.makedirs('results', exist_ok=True)
os.makedirs('graphs',  exist_ok=True)

# ── Plot style ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.framealpha': 0.9,
    'figure.facecolor': 'white',
})

COLORS = {
    'bb84':   '#2196F3',
    'e91':    '#9C27B0',
    'dh':     '#4CAF50',
    'ecc':    '#FF9800',
    'rsa':    '#F44336',
    'aes128': '#795548',
    'aes256': '#009688',
    'sha256': '#607D8B',
    'eve':    '#E53935',
    'no_eve': '#43A047',
}

ITERATIONS = 50  # statistical iterations per config

print("=" * 60)
print("  Protocol Simulation — Cryptographic Security Analysis")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
#  PROTOCOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

# ── Qiskit availability check ─────────────────────────────────
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from fractions import Fraction
    QISKIT_OK = True
    _sim = AerSimulator()
    print("  Qiskit AerSimulator: OK")
except Exception as e:
    QISKIT_OK = False
    print(f"  Qiskit not available ({e}) — using math fallback for BB84/E91")

try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_OK = True
    print("  cryptography library: OK")
except Exception as e:
    CRYPTO_OK = False
    print(f"  cryptography library not available ({e})")

print()

# ── BB84 ──────────────────────────────────────────────────────
# Source: Bennett & Brassard (1984/2011); Kushwah et al. (2023);
#         Quantum Blockchains (2022); Adu-Kyere et al. (2022)
def sim_bb84(n_bits, seed, eve_active=False):
    np.random.seed(seed)
    random.seed(seed)
    alice_bits  = np.random.randint(2, size=n_bits).tolist()
    alice_bases = np.random.randint(2, size=n_bits).tolist()
    bob_bases   = np.random.randint(2, size=n_bits).tolist()
    eve_bases   = np.random.randint(2, size=n_bits).tolist()

    t0 = time.perf_counter()

    if QISKIT_OK:
        bob_results = []
        for i in range(n_bits):
            qc = QuantumCircuit(1, 1)
            eff = alice_bits[i]
            if eve_active and eve_bases[i] != alice_bases[i]:
                eff = random.randint(0, 1)
            if alice_bases[i] == 0:
                if eff == 1: qc.x(0)
            else:
                if eff == 0: qc.h(0)
                else:        qc.x(0); qc.h(0)
            if bob_bases[i] == 1: qc.h(0)
            qc.measure(0, 0)
            job = _sim.run(transpile(qc, _sim), shots=1)
            bob_results.append(int(list(job.result().get_counts().keys())[0]))
    else:
        bob_results = []
        for i in range(n_bits):
            eff = alice_bits[i]
            if eve_active and eve_bases[i] != alice_bases[i]:
                eff = random.randint(0, 1)
            bob_results.append(eff if alice_bases[i] == bob_bases[i] else random.randint(0, 1))

    elapsed = (time.perf_counter() - t0) * 1000
    matches   = [alice_bases[i] == bob_bases[i] for i in range(n_bits)]
    alice_key = [alice_bits[i]   for i in range(n_bits) if matches[i]]
    bob_key   = [bob_results[i]  for i in range(n_bits) if matches[i]]
    errors    = sum(a != b for a, b in zip(alice_key, bob_key))
    qber      = errors / len(alice_key) * 100 if alice_key else 0.0
    return {
        'key_length': len(alice_key),
        'efficiency': len(alice_key) / n_bits * 100,
        'qber': qber,
        'errors': errors,
        'time_ms': elapsed,
    }

# ── E91 ───────────────────────────────────────────────────────
# Source: Ekert (1991); Begimbayeva & Zhaxalykov (2022);
#         Clauser, Horne, Shimony & Holt (1969) — CHSH inequality
def sim_e91(n_pairs, seed, eve_active=False):
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()

    # Alice: angles 0, 45, 90 deg  |  Bob: 45, 90, 135 deg
    alice_angles = [random.choice([0, 45, 90])   for _ in range(n_pairs)]
    bob_angles   = [random.choice([45, 90, 135]) for _ in range(n_pairs)]

    correlations = {(a, b): [] for a in [0,45,90] for b in [45,90,135]}
    alice_key, bob_key = [], []

    for i in range(n_pairs):
        a_ang, b_ang = alice_angles[i], bob_angles[i]
        a_bit = random.randint(0, 1)

        if eve_active:
            # Eve's intercept-resend measurement destroys entanglement —
            # Bob's result becomes fully uncorrelated with Alice's
            b_bit = random.randint(0, 1)
        else:
            # Quantum mechanical correlation for entangled qubits measured
            # at angles a_ang and b_ang: E(a,b) = -cos(a_ang - b_ang).
            # At matching angles (a_ang==b_ang), E=-1 → perfect
            # anti-correlation (b_bit = 1-a_bit), consistent with the
            # singlet-state assumption used for key sifting. At the four
            # CHSH test angle-pairs, this formula reproduces the maximal
            # Tsirelson bound S=2sqrt(2) (Clauser, Horne, Shimony & Holt,
            # 1969; Ekert, 1991).
            theta = math.radians(a_ang - b_ang)
            e_ab = -math.cos(theta)
            p_same = (1 + e_ab) / 2  # probability Bob's bit equals Alice's
            b_bit = a_bit if random.random() < p_same else 1 - a_bit

        # Record correlation for CHSH: +1 if same, -1 if different
        correlations[(a_ang, b_ang)].append(1 if a_bit == b_bit else -1)

        # Sifting: same angle → keep as key bit
        if a_ang == b_ang:
            alice_key.append(a_bit)
            bob_key.append(b_bit)

    # CHSH parameter S = E(0,45) - E(0,135) + E(90,45) + E(90,135)
    def E(a, b):
        vals = correlations.get((a, b), [])
        return np.mean(vals) if vals else 0.0

    S = abs(E(0,45) - E(0,135) + E(90,45) + E(90,135))

    elapsed  = (time.perf_counter() - t0) * 1000
    errors   = sum(a != b for a, b in zip(alice_key, bob_key))
    qber     = errors / len(alice_key) * 100 if alice_key else 0.0
    return {
        'key_length': len(alice_key),
        'efficiency': len(alice_key) / n_pairs * 100,
        'chsh_s': round(S, 4),
        'qber': qber,
        'time_ms': elapsed,
    }

# ── Diffie-Hellman ────────────────────────────────────────────
# Source: Diffie & Hellman (1976); Shanks (1971) — BSGS attack
DH_PRIMES = [23, 47, 59, 83, 107, 167, 179, 227, 263, 347]

def sim_dh(prime_idx, seed):
    random.seed(seed)
    p = DH_PRIMES[prime_idx % len(DH_PRIMES)]
    g = 2
    a = random.randint(2, p - 2)
    b = random.randint(2, p - 2)
    t0 = time.perf_counter()
    A = pow(g, a, p);  B = pow(g, b, p)
    K_a = pow(B, a, p);  K_b = pow(A, b, p)
    elapsed = (time.perf_counter() - t0) * 1000
    # BSGS attack
    t1 = time.perf_counter()
    m = math.ceil(math.sqrt(p))
    table = {}; val = 1; steps = 0
    for j in range(m):
        table[val] = j; val = (val * g) % p; steps += 1
    factor = pow(g, m * (p - 2), p); val = A; recovered = None
    for i in range(m):
        if val in table:
            recovered = i * m + table[val]; break
        val = (val * factor) % p; steps += 1
    atk_time = (time.perf_counter() - t1) * 1000
    return {
        'prime': p, 'key_bits': p.bit_length(),
        'time_ms': elapsed,
        'attack_steps': steps, 'attack_time_ms': atk_time,
        'keys_match': K_a == K_b, 'attack_success': recovered == a,
    }

# ── ECC ───────────────────────────────────────────────────────
# Source: Miller (1986); Koblitz (1987); brute-force ECDLP — standard method
class EllipticCurve:
    def __init__(self, a, b, p):
        self.a = a; self.b = b; self.p = p
    def point_add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        if P[0] == Q[0]:
            if P[1] != Q[1]: return None
            lam = (3*P[0]**2 + self.a) * pow(2*P[1], self.p-2, self.p) % self.p
        else:
            lam = (Q[1]-P[1]) * pow(Q[0]-P[0], self.p-2, self.p) % self.p
        x = (lam**2 - P[0] - Q[0]) % self.p
        y = (lam*(P[0] - x) - P[1]) % self.p
        return (x, y)
    def scalar_mult(self, k, P):
        result = None; addend = P
        while k:
            if k & 1: result = self.point_add(result, addend)
            addend = self.point_add(addend, addend); k >>= 1
        return result

def sim_ecc(seed):
    random.seed(seed)
    curve = EllipticCurve(2, 3, 97); G = (3, 6); n = 97
    ka = random.randint(2, n-2); kb = random.randint(2, n-2)
    t0 = time.perf_counter()
    Pa = curve.scalar_mult(ka, G); Pb = curve.scalar_mult(kb, G)
    Sa = curve.scalar_mult(ka, Pb); Sb = curve.scalar_mult(kb, Pa)
    elapsed = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter(); pt = G; steps = 0; recovered = None
    for k in range(1, n):
        if pt == Pa: recovered = k; break
        pt = curve.point_add(pt, G); steps += 1
    atk_time = (time.perf_counter() - t1) * 1000
    return {
        'key_bits': n.bit_length(), 'time_ms': elapsed,
        'attack_steps': steps, 'attack_time_ms': atk_time,
        'keys_match': Sa == Sb, 'attack_success': recovered == ka,
    }

# ── RSA ───────────────────────────────────────────────────────
# Source: Rivest, Shamir & Adleman (1978); Python cryptography library (PyCA, 2024)
def sim_rsa(key_size, seed):
    random.seed(seed)
    if not CRYPTO_OK:
        return {'key_bits': key_size, 'keygen_ms': 0.0, 'enc_ms': 0.0, 'dec_ms': 0.0}
    be = default_backend()
    t0 = time.perf_counter()
    priv = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=be)
    keygen_ms = (time.perf_counter() - t0) * 1000
    pub = priv.public_key()
    msg = b'X' * min(32, key_size // 8 - 66)
    t1 = time.perf_counter()
    ct = pub.encrypt(msg, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    enc_ms = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    priv.decrypt(ct, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    dec_ms = (time.perf_counter() - t2) * 1000
    return {'key_bits': key_size, 'keygen_ms': keygen_ms, 'enc_ms': enc_ms, 'dec_ms': dec_ms}

# ── AES ───────────────────────────────────────────────────────
# Source: NIST FIPS 197 (2001); Python cryptography library (PyCA, 2024)
# Grover's impact: Mavroeidis et al. (2018); Grover (1996)
def sim_aes(key_bits, seed, plaintext_size=1024):
    random.seed(seed)
    if not CRYPTO_OK:
        return {'key_bits': key_bits, 'enc_ms': 0.0, 'dec_ms': 0.0,
                'classical_security': key_bits, 'post_grover_security': key_bits // 2}
    key   = os.urandom(key_bits // 8)
    iv    = os.urandom(16)
    plain = os.urandom(plaintext_size)
    be    = default_backend()
    t0 = time.perf_counter()
    c = Cipher(algorithms.AES(key), modes.CBC(iv), backend=be)
    enc = c.encryptor()
    pad_len = 16 - (plaintext_size % 16)
    ct = enc.update(plain + bytes([pad_len] * pad_len)) + enc.finalize()
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    d = Cipher(algorithms.AES(key), modes.CBC(iv), backend=be).decryptor()
    d.update(ct) + d.finalize()
    dec_ms = (time.perf_counter() - t1) * 1000
    return {
        'key_bits': key_bits, 'enc_ms': enc_ms, 'dec_ms': dec_ms,
        'classical_security': key_bits,
        'post_grover_security': key_bits // 2,
    }

# ── SHA-256 ───────────────────────────────────────────────────
# Source: NIST FIPS 180-4 (2015); Python standard library hashlib
# Grover's impact: Mavroeidis et al. (2018); Grover (1996)
def sim_sha256(input_size, seed):
    random.seed(seed)
    data = os.urandom(input_size)
    t0 = time.perf_counter()
    hashlib.sha256(data).hexdigest()
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        'input_bytes': input_size,
        'hash_ms': elapsed,
        'classical_preimage_bits': 256,
        'post_grover_preimage_bits': 128,
        'classical_collision_bits': 128,
        'post_grover_collision_bits': 64,
    }


# ══════════════════════════════════════════════════════════════
#  CSV HELPER
# ══════════════════════════════════════════════════════════════
def save_csv(filename, rows):
    if not rows: return
    path = os.path.join('results', filename)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: results/{filename}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — BB84 Efficiency vs Qubit Count
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 1] BB84 efficiency vs qubit count...")
QUBIT_COUNTS = [8, 16, 32, 64, 128, 256, 512]
exp1_rows = []
exp1_means = {'n': [], 'key_len': [], 'efficiency': [], 'time_ms': []}
exp1_stds  = {'key_len': [], 'efficiency': [], 'time_ms': []}

for n in QUBIT_COUNTS:
    kl, ef, tm = [], [], []
    for s in range(ITERATIONS):
        r = sim_bb84(n, s, eve_active=False)
        kl.append(r['key_length']); ef.append(r['efficiency']); tm.append(r['time_ms'])
        exp1_rows.append({'protocol':'BB84','n_qubits':n,'seed':s,
                          'key_length':r['key_length'],'efficiency':r['efficiency'],
                          'time_ms':round(r['time_ms'],3)})
    exp1_means['n'].append(n)
    exp1_means['key_len'].append(np.mean(kl));  exp1_stds['key_len'].append(np.std(kl))
    exp1_means['efficiency'].append(np.mean(ef));exp1_stds['efficiency'].append(np.std(ef))
    exp1_means['time_ms'].append(np.mean(tm));   exp1_stds['time_ms'].append(np.std(tm))
    print(f"  n={n:4d}  key={np.mean(kl):.1f}±{np.std(kl):.1f}  eff={np.mean(ef):.1f}%  t={np.mean(tm):.1f}ms")

save_csv('exp1_bb84_efficiency.csv', exp1_rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Experiment 1 — BB84 Key Generation Efficiency vs Qubit Count\n(50 iterations per configuration, Qiskit AerSimulator)', y=1.02)

axes[0].errorbar(exp1_means['n'], exp1_means['key_len'], yerr=exp1_stds['key_len'],
                 color=COLORS['bb84'], marker='o', linewidth=2, capsize=4, label='Mean ± SD')
axes[0].plot(exp1_means['n'], [n*0.5 for n in exp1_means['n']], 'k--', alpha=0.5, label='Theoretical 50%')
axes[0].set_xlabel('Number of Qubits (n)'); axes[0].set_ylabel('Sifted Key Length (bits)')
axes[0].set_title('Sifted Key Length vs Qubit Count'); axes[0].legend(); axes[0].set_xscale('log', base=2)

axes[1].errorbar(exp1_means['n'], exp1_means['efficiency'], yerr=exp1_stds['efficiency'],
                 color=COLORS['bb84'], marker='s', linewidth=2, capsize=4)
axes[1].axhline(50, color='k', linestyle='--', alpha=0.5, label='Theoretical 50%')
axes[1].set_xlabel('Number of Qubits (n)'); axes[1].set_ylabel('Sifting Efficiency (%)')
axes[1].set_title('Key Sifting Efficiency vs Qubit Count'); axes[1].legend(); axes[1].set_xscale('log', base=2)
axes[1].set_ylim(0, 100)

axes[2].errorbar(exp1_means['n'], exp1_means['time_ms'], yerr=exp1_stds['time_ms'],
                 color=COLORS['bb84'], marker='^', linewidth=2, capsize=4)
axes[2].set_xlabel('Number of Qubits (n)'); axes[2].set_ylabel('Generation Time (ms)')
axes[2].set_title('Key Generation Time vs Qubit Count'); axes[2].set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('graphs/exp1_bb84_efficiency.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp1_bb84_efficiency.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — BB84 QBER Analysis
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 2] BB84 QBER analysis with and without Eve...")
QBER_QUBITS = [8, 16, 32, 64, 128]
exp2_rows = []
exp2_qber_eve, exp2_qber_noeve = [], []
exp2_qber_eve_std, exp2_qber_noeve_std = [], []

for n in QBER_QUBITS:
    qber_e, qber_ne = [], []
    for s in range(ITERATIONS):
        r_e  = sim_bb84(n, s, eve_active=True)
        r_ne = sim_bb84(n, s, eve_active=False)
        qber_e.append(r_e['qber']); qber_ne.append(r_ne['qber'])
        exp2_rows.append({'protocol':'BB84','n_qubits':n,'seed':s,'eve':True,
                          'qber':round(r_e['qber'],3),'key_length':r_e['key_length']})
        exp2_rows.append({'protocol':'BB84','n_qubits':n,'seed':s,'eve':False,
                          'qber':round(r_ne['qber'],3),'key_length':r_ne['key_length']})
    exp2_qber_eve.append(np.mean(qber_e));    exp2_qber_eve_std.append(np.std(qber_e))
    exp2_qber_noeve.append(np.mean(qber_ne)); exp2_qber_noeve_std.append(np.std(qber_ne))
    print(f"  n={n:4d}  QBER(Eve)={np.mean(qber_e):.1f}%±{np.std(qber_e):.1f}  QBER(no Eve)={np.mean(qber_ne):.1f}%")

save_csv('exp2_bb84_qber.csv', exp2_rows)

fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(QBER_QUBITS, exp2_qber_eve, yerr=exp2_qber_eve_std,
            color=COLORS['eve'], marker='o', linewidth=2, capsize=5, label='With Eve (eavesdropper)')
ax.errorbar(QBER_QUBITS, exp2_qber_noeve, yerr=exp2_qber_noeve_std,
            color=COLORS['no_eve'], marker='s', linewidth=2, capsize=5, label='Without Eve')
ax.axhline(25, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='25% QBER detection threshold')
ax.set_xlabel('Number of Qubits (n)'); ax.set_ylabel('QBER (%)')
ax.set_title('Experiment 2 — BB84 QBER Analysis\nWith and Without Eavesdropper (50 iterations per configuration)')
ax.legend(); ax.set_ylim(-2, 55)
ax.fill_between(QBER_QUBITS, 25, 55, alpha=0.06, color='red', label='Eavesdropping detected zone')
plt.tight_layout()
plt.savefig('graphs/exp2_bb84_qber.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp2_bb84_qber.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — E91 CHSH Analysis
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 3] E91 CHSH parameter analysis...")
exp3_rows = []
exp3_s_eve, exp3_s_noeve = [], []
exp3_s_eve_std, exp3_s_noeve_std = [], []
CHSH_IDEAL = 2 * math.sqrt(2)

for n in QBER_QUBITS:
    s_e, s_ne = [], []
    for s in range(ITERATIONS):
        r_e  = sim_e91(n, s, eve_active=True)
        r_ne = sim_e91(n, s, eve_active=False)
        s_e.append(r_e['chsh_s']); s_ne.append(r_ne['chsh_s'])
        exp3_rows.append({'protocol':'E91','n_pairs':n,'seed':s,'eve':True,
                          'chsh_s':r_e['chsh_s'],'key_length':r_e['key_length'],'efficiency':round(r_e['efficiency'],2)})
        exp3_rows.append({'protocol':'E91','n_pairs':n,'seed':s,'eve':False,
                          'chsh_s':r_ne['chsh_s'],'key_length':r_ne['key_length'],'efficiency':round(r_ne['efficiency'],2)})
    exp3_s_eve.append(np.mean(s_e));    exp3_s_eve_std.append(np.std(s_e))
    exp3_s_noeve.append(np.mean(s_ne)); exp3_s_noeve_std.append(np.std(s_ne))
    print(f"  n={n:4d}  S(Eve)={np.mean(s_e):.3f}±{np.std(s_e):.3f}  S(no Eve)={np.mean(s_ne):.3f}")

save_csv('exp3_e91_chsh.csv', exp3_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Experiment 3 — E91 CHSH Parameter Analysis\n(50 iterations per configuration)')

axes[0].errorbar(QBER_QUBITS, exp3_s_noeve, yerr=exp3_s_noeve_std,
                 color=COLORS['e91'], marker='o', linewidth=2, capsize=5, label='Without Eve')
axes[0].errorbar(QBER_QUBITS, exp3_s_eve, yerr=exp3_s_eve_std,
                 color=COLORS['eve'], marker='s', linewidth=2, capsize=5, label='With Eve')
axes[0].axhline(CHSH_IDEAL, color='green', linestyle='--', alpha=0.8, linewidth=1.5,
                label=f'Quantum bound 2√2 ≈ {CHSH_IDEAL:.3f}')
axes[0].axhline(2.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5,
                label='Classical limit S = 2')
axes[0].fill_between(QBER_QUBITS, 0, 2, alpha=0.06, color='red')
axes[0].set_xlabel('Number of Entangled Pairs'); axes[0].set_ylabel('CHSH Parameter S')
axes[0].set_title('CHSH Parameter S vs Entangled Pairs'); axes[0].legend()

# E91 vs BB84 efficiency comparison
bb84_eff_64 = [r['efficiency'] for r in exp1_rows if r['n_qubits'] == 64]
e91_eff_64  = [r['efficiency'] for r in exp3_rows if r['n_pairs'] == 64 and not r['eve']]
axes[1].boxplot([bb84_eff_64, e91_eff_64], labels=['BB84\n(n=64)', 'E91\n(n=64)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='navy', linewidth=2))
axes[1].axhline(50, color=COLORS['bb84'], linestyle='--', alpha=0.7, label='BB84 theoretical 50%')
axes[1].axhline(22.2, color=COLORS['e91'],  linestyle='--', alpha=0.7, label='E91 theoretical 2/9≈22.2%')
axes[1].set_ylabel('Sifting Efficiency (%)'); axes[1].set_title('Key Efficiency: BB84 vs E91 (n=64)')
axes[1].legend()

plt.tight_layout()
plt.savefig('graphs/exp3_e91_chsh.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp3_e91_chsh.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — DH and ECC Attack Complexity
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 4] DH and ECC attack complexity...")
exp4_rows = []
dh_means = {'prime': [], 'bits': [], 'steps': [], 'time': []}
dh_stds  = {'steps': [], 'time': []}

for idx in range(len(DH_PRIMES)):
    steps_list, time_list = [], []
    for s in range(ITERATIONS):
        r = sim_dh(idx, s)
        steps_list.append(r['attack_steps']); time_list.append(r['attack_time_ms'])
        exp4_rows.append({'protocol':'DH','prime_idx':idx,'prime':r['prime'],
                          'key_bits':r['key_bits'],'seed':s,
                          'attack_steps':r['attack_steps'],'attack_time_ms':round(r['attack_time_ms'],4),
                          'keygen_time_ms':round(r['time_ms'],4)})
    dh_means['prime'].append(DH_PRIMES[idx]); dh_means['bits'].append(DH_PRIMES[idx].bit_length())
    dh_means['steps'].append(np.mean(steps_list)); dh_stds['steps'].append(np.std(steps_list))
    dh_means['time'].append(np.mean(time_list));   dh_stds['time'].append(np.std(time_list))
    print(f"  DH p={DH_PRIMES[idx]:4d}  steps={np.mean(steps_list):.0f}±{np.std(steps_list):.0f}")

ecc_steps_list, ecc_time_list = [], []
for s in range(ITERATIONS):
    r = sim_ecc(s)
    ecc_steps_list.append(r['attack_steps']); ecc_time_list.append(r['attack_time_ms'])
    exp4_rows.append({'protocol':'ECC','prime_idx':0,'prime':97,'key_bits':r['key_bits'],'seed':s,
                      'attack_steps':r['attack_steps'],'attack_time_ms':round(r['attack_time_ms'],4),
                      'keygen_time_ms':round(r['time_ms'],4)})
print(f"  ECC p=97   steps={np.mean(ecc_steps_list):.0f}±{np.std(ecc_steps_list):.0f}")
save_csv('exp4_dh_ecc_attack.csv', exp4_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Experiment 4 — DH & ECC Classical Attack Complexity vs Key Size\n(50 iterations; Baby-step Giant-step for DH, Brute-force ECDLP for ECC)')

axes[0].errorbar(dh_means['prime'], dh_means['steps'], yerr=dh_stds['steps'],
                 color=COLORS['dh'], marker='o', linewidth=2, capsize=5, label='BSGS steps (DH)')
sqrt_p = [math.sqrt(p) for p in dh_means['prime']]
axes[0].plot(dh_means['prime'], sqrt_p, 'k--', alpha=0.6, label='O(√p) theoretical')
axes[0].set_xlabel('Prime Modulus p'); axes[0].set_ylabel('Attack Steps')
axes[0].set_title('DH: BSGS Attack Steps vs Prime Size'); axes[0].legend()

categories = [f'p={p}' for p in DH_PRIMES] + ['ECC\n(p=97)']
all_steps  = dh_means['steps'] + [np.mean(ecc_steps_list)]
bar_colors = [COLORS['dh']] * len(DH_PRIMES) + [COLORS['ecc']]
bars = axes[1].bar(categories, all_steps, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Protocol / Prime'); axes[1].set_ylabel('Mean Attack Steps')
axes[1].set_title('Attack Steps: DH (BSGS) vs ECC (Brute-force)')
axes[1].tick_params(axis='x', rotation=30)
patch_dh  = mpatches.Patch(color=COLORS['dh'],  label='Diffie-Hellman (BSGS)')
patch_ecc = mpatches.Patch(color=COLORS['ecc'], label='ECC (Brute-force ECDLP)')
axes[1].legend(handles=[patch_dh, patch_ecc])

plt.tight_layout()
plt.savefig('graphs/exp4_dh_ecc_attack.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp4_dh_ecc_attack.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 5 — RSA Key Generation vs Key Size
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 5] RSA key generation time vs key size...")
RSA_SIZES = [1024, 2048]
exp5_rows = []
rsa_kg_means, rsa_kg_stds = [], []
rsa_enc_means, rsa_dec_means = [], []

# Warm-up: discard first encrypt call to avoid cold-start artefact
if CRYPTO_OK:
    _wu_priv = rsa.generate_private_key(
        public_exponent=65537, key_size=1024, backend=default_backend())
    _wu_priv.public_key().encrypt(
        b'warmup',
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None))

for ks in RSA_SIZES:
    kg, enc, dec = [], [], []
    iters = min(ITERATIONS, 20)  # RSA is slow at 2048-bit
    for s in range(iters):
        r = sim_rsa(ks, s)
        kg.append(r['keygen_ms']); enc.append(r['enc_ms']); dec.append(r['dec_ms'])
        exp5_rows.append({'protocol':'RSA','key_bits':ks,'seed':s,
                          'keygen_ms':round(r['keygen_ms'],3),
                          'enc_ms':round(r['enc_ms'],3),'dec_ms':round(r['dec_ms'],3)})
    rsa_kg_means.append(np.mean(kg));  rsa_kg_stds.append(np.std(kg))
    rsa_enc_means.append(np.mean(enc)); rsa_dec_means.append(np.mean(dec))
    print(f"  RSA-{ks:4d}  keygen={np.mean(kg):.1f}ms  enc={np.mean(enc):.2f}ms  dec={np.mean(dec):.2f}ms")

save_csv('exp5_rsa_keygen.csv', exp5_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Experiment 5 — RSA Key Generation & Operation Time vs Key Size\n(20 iterations per configuration)')

axes[0].bar([str(s) for s in RSA_SIZES], rsa_kg_means,
            yerr=rsa_kg_stds, color=COLORS['rsa'], capsize=5, edgecolor='white')
axes[0].set_xlabel('RSA Key Size (bits)'); axes[0].set_ylabel('Key Generation Time (ms)')
axes[0].set_title('RSA Key Generation Time vs Key Size')

x = np.arange(len(RSA_SIZES)); w = 0.35
axes[1].bar(x - w/2, rsa_enc_means, w, label='Encryption', color=COLORS['rsa'], alpha=0.8)
axes[1].bar(x + w/2, rsa_dec_means, w, label='Decryption', color='#B71C1C', alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([str(s) for s in RSA_SIZES])
axes[1].set_xlabel('RSA Key Size (bits)'); axes[1].set_ylabel('Operation Time (ms)')
axes[1].set_title('RSA Encryption vs Decryption Time'); axes[1].legend()

plt.tight_layout()
plt.savefig('graphs/exp5_rsa_keygen.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp5_rsa_keygen.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 6 — AES-128 vs AES-256 + Grover's Impact
# ══════════════════════════════════════════════════════════════
def _clip_yerr(means, stds):
    """Return asymmetric error bars that never extend below zero. Timing values
    are non-negative, so an error bar dipping below 0 is physically meaningless;
    the lower whisker is capped at the mean while the upper keeps the full std."""
    means = np.asarray(means, dtype=float)
    stds  = np.asarray(stds,  dtype=float)
    lower = np.minimum(stds, means)
    return np.vstack([lower, stds])


print("\n[Experiment 6] AES-128 vs AES-256 performance and Grover's threat...")
AES_SIZES   = [128, 256]
PLAIN_SIZES = [128, 512, 1024]
exp6_rows = []
aes_enc_means = {128: [], 256: []}
aes_enc_stds  = {128: [], 256: []}

# Warm-up: one discarded AES call so the OpenSSL backend pays its one-time
# initialization cost OUTSIDE the timed region. Without this, the first timed
# configuration (AES-128, 128B, seed 0) absorbs ~0.4ms of backend init,
# inflating that single data point's mean and standard deviation.
if CRYPTO_OK:
    _wu_key = os.urandom(16); _wu_iv = os.urandom(16)
    _wu_c = Cipher(algorithms.AES(_wu_key), modes.CBC(_wu_iv), backend=default_backend())
    _ = _wu_c.encryptor().update(b'\x00' * 16 + b'\x10' * 16)

for ps in PLAIN_SIZES:
    for ks in AES_SIZES:
        enc_list = []
        for s in range(ITERATIONS):
            r = sim_aes(ks, s, ps)
            enc_list.append(r['enc_ms'])
            exp6_rows.append({'protocol':f'AES-{ks}','key_bits':ks,'plaintext_bytes':ps,'seed':s,
                              'enc_ms':round(r['enc_ms'],4),'dec_ms':round(r['dec_ms'],4),
                              'classical_security':r['classical_security'],
                              'post_grover_security':r['post_grover_security']})
        aes_enc_means[ks].append(np.mean(enc_list))
        aes_enc_stds[ks].append(np.std(enc_list))
        print(f"  AES-{ks}  plaintext={ps:5d}B  enc={np.mean(enc_list):.4f}ms")

save_csv('exp6_aes_performance.csv', exp6_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Experiment 6 — AES-128 vs AES-256: Performance & Quantum Security\n(Grover\'s Algorithm Impact on Effective Security Bits)')

x = np.arange(len(PLAIN_SIZES)); w = 0.35
axes[0].bar(x - w/2, aes_enc_means[128], w, yerr=_clip_yerr(aes_enc_means[128], aes_enc_stds[128]),
            label='AES-128', color=COLORS['aes128'], capsize=4, alpha=0.9)
axes[0].bar(x + w/2, aes_enc_means[256], w, yerr=_clip_yerr(aes_enc_means[256], aes_enc_stds[256]),
            label='AES-256', color=COLORS['aes256'], capsize=4, alpha=0.9)
axes[0].set_ylim(bottom=0)
axes[0].set_xticks(x); axes[0].set_xticklabels([f'{p}B' for p in PLAIN_SIZES])
axes[0].set_xlabel('Plaintext Size'); axes[0].set_ylabel('Encryption Time (ms)')
axes[0].set_title('AES Encryption Time vs Plaintext Size'); axes[0].legend()

protocols = ['AES-128\n(Classical)', 'AES-128\n(Post-Grover)', 'AES-256\n(Classical)', 'AES-256\n(Post-Grover)']
sec_bits  = [128, 64, 256, 128]
bar_cols  = [COLORS['aes128'], COLORS['eve'], COLORS['aes256'], COLORS['no_eve']]
bars = axes[1].bar(protocols, sec_bits, color=bar_cols, edgecolor='white', linewidth=0.5)
axes[1].axhline(128, color='blue',  linestyle='--', alpha=0.6, linewidth=1.5, label='128-bit safety threshold')
axes[1].axhline(80,  color='orange',linestyle='--', alpha=0.6, linewidth=1.5, label='80-bit minimum threshold')
axes[1].set_ylabel('Effective Security Level (bits)')
axes[1].set_title("Security Bits: Classical vs Post-Grover's")
axes[1].legend(fontsize=9)
for bar, val in zip(bars, sec_bits):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f'{val}-bit',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/exp6_aes_grover.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp6_aes_grover.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 7 — SHA-256 Performance + Grover's Impact
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 7] SHA-256 hashing performance and Grover's impact...")
SHA_SIZES = [64, 256, 1024, 4096, 16384]
exp7_rows = []
sha_means, sha_stds = [], []

# Warm-up: one discarded hashlib.sha256 call so the module pays its one-time
# initialization cost OUTSIDE the timed region. Without this, the first timed
# input size (64 bytes, seed 0) is inflated ~35x (true ~1.5us measured as
# ~53us), producing a standard deviation so large the error bar dips below 0.
hashlib.sha256(b'\x00' * 64).hexdigest()

for sz in SHA_SIZES:
    times = []
    for s in range(ITERATIONS):
        r = sim_sha256(sz, s)
        times.append(r['hash_ms'])
        exp7_rows.append({'protocol':'SHA-256','input_bytes':sz,'seed':s,
                          'hash_ms':round(r['hash_ms'],5),
                          'classical_preimage_bits':256,'post_grover_preimage_bits':128,
                          'classical_collision_bits':128,'post_grover_collision_bits':64})
    sha_means.append(np.mean(times)); sha_stds.append(np.std(times))
    print(f"  SHA-256  input={sz:6d}B  time={np.mean(times)*1000:.3f}µs")

save_csv('exp7_sha256_performance.csv', exp7_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Experiment 7 — SHA-256 Performance & Grover's Algorithm Impact\non Hash Security Properties")

axes[0].errorbar(SHA_SIZES, [m*1000 for m in sha_means],
                 yerr=_clip_yerr([m*1000 for m in sha_means], [s*1000 for s in sha_stds]),
                 color=COLORS['sha256'], marker='o', linewidth=2, capsize=4)
axes[0].set_xlabel('Input Size (bytes)'); axes[0].set_ylabel('Hash Computation Time (µs)')
axes[0].set_title('SHA-256 Computation Time vs Input Size'); axes[0].set_xscale('log', base=2)

categories = ['Preimage\n(Classical)', "Preimage\n(Post-Grover's)", 'Collision\n(Classical)', "Collision\n(Post-Grover's)"]
values  = [256, 128, 128, 64]
colors_ = [COLORS['sha256'], COLORS['eve'], '#455A64', '#E53935']
bars = axes[1].bar(categories, values, color=colors_, edgecolor='white', linewidth=0.5)
axes[1].axhline(128, color='blue',   linestyle='--', alpha=0.6, linewidth=1.5, label='128-bit safety threshold')
axes[1].axhline(80,  color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label='80-bit minimum threshold')
axes[1].set_ylabel('Security Level (bits)')
axes[1].set_title("SHA-256 Security Bits: Classical vs Post-Grover's")
axes[1].legend(fontsize=9)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/exp7_sha256_grover.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp7_sha256_grover.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 8 — Cross-Protocol Comparison
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 8] Cross-protocol comparison (all 8 protocols)...")

# Gather fixed-param means across 50 seeds
def collect(fn, n=ITERATIONS):
    results = [fn(s) for s in range(n)]
    return results

bb84_64  = collect(lambda s: sim_bb84(64, s, False))
e91_64   = collect(lambda s: sim_e91(64, s, False))
dh_std   = collect(lambda s: sim_dh(9, s))       # p=347 (largest)
ecc_std  = collect(lambda s: sim_ecc(s))
rsa_std  = collect(lambda s: sim_rsa(2048, s))[:20]  # 20 iters for RSA
aes128   = collect(lambda s: sim_aes(128, s, 1024))
aes256   = collect(lambda s: sim_aes(256, s, 1024))
sha256   = collect(lambda s: sim_sha256(1024, s))

protocols_cmp = ['BB84', 'E91', 'DH', 'ECC', 'RSA-2048', 'AES-128', 'AES-256', 'SHA-256']
col_map  = [COLORS['bb84'], COLORS['e91'], COLORS['dh'], COLORS['ecc'],
            COLORS['rsa'], COLORS['aes128'], COLORS['aes256'], COLORS['sha256']]

gen_times = [
    np.mean([r['time_ms'] for r in bb84_64]),
    np.mean([r['time_ms'] for r in e91_64]),
    np.mean([r['time_ms'] for r in dh_std]),
    np.mean([r['time_ms'] for r in ecc_std]),
    np.mean([r['keygen_ms'] for r in rsa_std]),
    np.mean([r['enc_ms'] for r in aes128]),
    np.mean([r['enc_ms'] for r in aes256]),
    np.mean([r['hash_ms'] for r in sha256]),
]

# Security bits under quantum threat
# NOTE: DH and ECC use bit-length of their key/curve parameter (key_bits),
# matching the same unit used for RSA/AES/SHA, rather than the raw
# prime/modulus value. This demonstration uses small primes (p=347, mod 97)
# for tractable BSGS/ECDLP attacks, so their bit-length is intentionally
# much smaller than real-world deployments (DH/ECC normally use 2048+/256-bit
# keys). See Limitations section for discussion of this scale difference.
sec_classical = [128, 64, dh_std[0]['key_bits'], ecc_std[0]['key_bits'], 2048, 128, 256, 256]
sec_quantum   = [128, 64,   0,  0,    0,  64, 128, 64]
threat_labels = ['Quantum-native\nsecure', 'Quantum-native\nsecure',
                 'Broken\n(Shor\'s)', 'Broken\n(Shor\'s)',
                 'Broken\n(Shor\'s)', 'Weakened\n(Grover\'s)',
                 'Quantum\nresistant', 'Weakened\n(Grover\'s)']

exp8_rows = []
for i, proto in enumerate(protocols_cmp):
    exp8_rows.append({
        'protocol': proto,
        'gen_time_ms_mean': round(gen_times[i], 4),
        'classical_security_bits': sec_classical[i],
        'post_quantum_security_bits': sec_quantum[i],
        'quantum_threat': threat_labels[i].replace('\n', ' '),
    })
save_csv('exp8_cross_protocol_comparison.csv', exp8_rows)

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Experiment 8 — Cross-Protocol Comparison: All 8 Cryptographic Protocols\n'
             'Quantum vs Classical Security Under Shor\'s and Grover\'s Algorithms', fontsize=14, y=1.01)
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1 — Generation time
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(protocols_cmp, gen_times, color=col_map, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Mean Generation/Operation Time (ms)')
ax1.set_title('Key Generation / Operation Time')
for bar, val in zip(bars, gen_times):
    ax1.text(bar.get_width() + max(gen_times)*0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}ms', va='center', fontsize=9)
ax1.set_xlim(0, max(gen_times) * 1.25)

# Plot 2 — Classical vs Quantum security bits
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(protocols_cmp)); w = 0.35
b1 = ax2.bar(x - w/2, sec_classical, w, label='Classical security', color=col_map, alpha=0.9)
b2 = ax2.bar(x + w/2, sec_quantum,   w, label='Post-quantum security', color=col_map, alpha=0.4, hatch='//')
ax2.axhline(128, color='blue',   linestyle='--', alpha=0.6, label='128-bit threshold')
ax2.axhline(80,  color='orange', linestyle='--', alpha=0.6, label='80-bit minimum')
ax2.set_xticks(x); ax2.set_xticklabels(protocols_cmp, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Security Level (bits)')
ax2.set_title('Classical vs Post-Quantum Security Bits')
ax2.legend(fontsize=8)

# Plot 3 — QBER BB84 vs E91 efficiency
ax3 = fig.add_subplot(gs[1, 0])
bb84_eff = [r['efficiency'] for r in bb84_64]
e91_eff  = [r['efficiency'] for r in e91_64]
ax3.hist(bb84_eff, bins=15, alpha=0.7, color=COLORS['bb84'], label=f'BB84 (mean={np.mean(bb84_eff):.1f}%)')
ax3.hist(e91_eff,  bins=15, alpha=0.7, color=COLORS['e91'],  label=f'E91  (mean={np.mean(e91_eff):.1f}%)')
ax3.axvline(50, color=COLORS['bb84'], linestyle='--', alpha=0.7)
ax3.axvline(22.2, color=COLORS['e91'],  linestyle='--', alpha=0.7)
ax3.set_xlabel('Sifting Efficiency (%)'); ax3.set_ylabel('Frequency')
ax3.set_title('BB84 vs E91 Key Sifting Efficiency Distribution (n=64)')
ax3.legend()

# Plot 4 — Quantum threat classification heat map
ax4 = fig.add_subplot(gs[1, 1])
threat_matrix = np.array([
    [0, 0],   # BB84:   Shor=safe, Grover=safe
    [0, 0],   # E91:    safe, safe
    [3, 0],   # DH:     broken, N/A
    [3, 0],   # ECC:    broken, N/A
    [3, 0],   # RSA:    broken, N/A
    [0, 2],   # AES128: N/A, severely weakened
    [0, 1],   # AES256: N/A, weakened but safe
    [0, 2],   # SHA256: N/A, weakened
])
im = ax4.imshow(threat_matrix.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=3)
ax4.set_xticks(range(len(protocols_cmp))); ax4.set_xticklabels(protocols_cmp, rotation=30, ha='right', fontsize=9)
ax4.set_yticks([0, 1]); ax4.set_yticklabels(["Shor's\nAlgorithm", "Grover's\nAlgorithm"])
ax4.set_title('Quantum Threat Classification Heatmap')
cell_text = [['Safe', 'Safe', 'BROKEN', 'BROKEN', 'BROKEN', 'N/A', 'N/A', 'N/A'],
             ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'Weakened', 'Resistant', 'Weakened']]
for row in range(2):
    for col in range(len(protocols_cmp)):
        txt = cell_text[row][col]
        color = 'white' if threat_matrix[col][row] == 3 else 'black'
        ax4.text(col, row, txt, ha='center', va='center', fontsize=8, fontweight='bold', color=color)

from matplotlib.colors import LinearSegmentedColormap
cbar = plt.colorbar(im, ax=ax4, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(['Safe/N/A', 'Weakened', 'Severely\nWeakened', 'Broken'])

plt.savefig('graphs/exp8_cross_protocol_comparison.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/exp8_cross_protocol_comparison.png")


# ══════════════════════════════════════════════════════════════
#  BONUS — BB84 vs E91 Head-to-Head Summary
# ══════════════════════════════════════════════════════════════
print("\n[Bonus] BB84 vs E91 head-to-head summary graph...")
ns = QBER_QUBITS
bb84_eff_n, e91_eff_n = [], []
bb84_qber_eve_n, e91_chsh_eve_n = [], []

for n in ns:
    b_eff = np.mean([sim_bb84(n, s, False)['efficiency'] for s in range(30)])
    e_eff = np.mean([sim_e91(n, s, False)['efficiency']  for s in range(30)])
    b_qber = np.mean([sim_bb84(n, s, True)['qber'] for s in range(30)])
    e_chsh = np.mean([sim_e91(n, s, True)['chsh_s'] for s in range(30)])
    bb84_eff_n.append(b_eff); e91_eff_n.append(e_eff)
    bb84_qber_eve_n.append(b_qber); e91_chsh_eve_n.append(e_chsh)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bonus — BB84 vs E91: Efficiency and Eavesdrop Detection Comparison\n(30 iterations per configuration)')

axes[0].plot(ns, bb84_eff_n, color=COLORS['bb84'], marker='o', linewidth=2, label='BB84 efficiency')
axes[0].plot(ns, e91_eff_n,  color=COLORS['e91'],  marker='s', linewidth=2, label='E91 efficiency')
axes[0].axhline(50, color=COLORS['bb84'], linestyle='--', alpha=0.5)
axes[0].axhline(22.2, color=COLORS['e91'],  linestyle='--', alpha=0.5)
axes[0].set_xlabel('Number of Qubits/Pairs'); axes[0].set_ylabel('Sifting Efficiency (%)')
axes[0].set_title('Key Sifting Efficiency: BB84 vs E91'); axes[0].legend(); axes[0].set_ylim(0, 80)

ax2b = axes[1].twinx()
axes[1].plot(ns, bb84_qber_eve_n, color=COLORS['bb84'], marker='o', linewidth=2, label='BB84 QBER with Eve (%)')
axes[1].axhline(25, color=COLORS['bb84'], linestyle='--', alpha=0.5, label='BB84 25% threshold')
ax2b.plot(ns, e91_chsh_eve_n, color=COLORS['e91'], marker='s', linewidth=2, linestyle='-.', label='E91 CHSH S with Eve')
ax2b.axhline(2.0, color=COLORS['e91'], linestyle=':', alpha=0.7, label='E91 classical limit S=2')
ax2b.axhline(CHSH_IDEAL, color='green', linestyle=':', alpha=0.5, label=f'E91 quantum bound {CHSH_IDEAL:.2f}')
axes[1].set_xlabel('Number of Qubits/Pairs')
axes[1].set_ylabel('BB84 QBER (%)', color=COLORS['bb84'])
ax2b.set_ylabel('E91 CHSH Parameter S', color=COLORS['e91'])
axes[1].set_title('Eavesdrop Detection: BB84 QBER vs E91 CHSH')
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('graphs/bonus_bb84_vs_e91.png', bbox_inches='tight')
plt.close()
print("  Graph saved: graphs/bonus_bb84_vs_e91.png")


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 9 — Shor's Algorithm (Qiskit QFT / Phase Estimation)
#  Quantum attack on Diffie-Hellman: order finding of a mod N
#  Uses real quantum gates — H, controlled-U, inverse QFT
#  Runs on small primes only (N ≤ 63) — limited by qubit count
# ══════════════════════════════════════════════════════════════
print("\n[Experiment 9] Shor's algorithm — Qiskit quantum order finding...")

def build_shors_circuit(a, N, n_count):
    """
    Build a faithful Shor's order-finding circuit using Quantum Phase
    Estimation on the modular-multiplication operator U|x> = |(a*x) mod N>.
    Finds r such that a^r ≡ 1 (mod N).

    Each controlled-U^(2^j) operation is implemented as an exact unitary
    (permutation) matrix that maps |x> -> |(a^(2^j) * x) mod N> for x < N and
    acts as the identity on the unused basis states x >= N. This is a genuine
    modular-exponentiation oracle — not a phase approximation — so the QPE
    phase register encodes true integer multiples of 1/r, and the continued-
    fractions post-processing recovers the exact multiplicative order.

    n_count = counting qubits (phase register)
    n_work  = work qubits = ceil(log2(N + 1))

    Source: Shor (1994, 1997); Nielsen & Chuang (2010) Ch.5;
            IBM Qiskit Documentation (2024) — QFT and Phase Estimation
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import QFT, UnitaryGate
    except ImportError:
        return None

    n_work = max(math.ceil(math.log2(N + 1)), 1)
    dim = 2 ** n_work
    qc = QuantumCircuit(n_count + n_work, n_count)

    # Initialise counting register to uniform superposition
    for q in range(n_count):
        qc.h(q)

    # Initialise work register to |1>
    qc.x(n_count)

    # Apply controlled-U^(2^j): U|x> = |(a*x) mod N>, built as an exact
    # permutation-matrix unitary for each repeated-squaring power a^(2^j).
    for j in range(n_count):
        a_pow = pow(a, 2 ** j, N)
        U = np.zeros((dim, dim))
        for x in range(dim):
            U[((a_pow * x) % N) if x < N else x, x] = 1.0
        cU = UnitaryGate(U, label=f"a^{2**j} mod N").control(1)
        qc.append(cU, [j] + list(range(n_count, n_count + n_work)))

    # Inverse QFT on counting register
    iqft = QFT(n_count, inverse=True, do_swaps=True).to_gate()
    qc.append(iqft, range(n_count))

    # Measure counting register
    qc.measure(range(n_count), range(n_count))
    return qc


def run_shors_order_finding(a, N, n_count=4, shots=1024):
    """
    Run Shor's order-finding on Qiskit AerSimulator.
    Returns (order r, top count, elapsed ms, transpiled circuit depth).

    The strongest measurement peaks are examined and the continued-fractions
    expansion of each measured phase is tested; the smallest valid order r
    satisfying a^r ≡ 1 (mod N) is returned. This mirrors the classical
    post-processing of Shor's algorithm, which inspects measurement outcomes
    to recover the true period r.
    """
    if not QISKIT_OK:
        return None, 0, 0.0, 0

    try:
        from qiskit import transpile
        from fractions import Fraction
    except ImportError:
        return None, 0, 0.0, 0

    qc = build_shors_circuit(a, N, n_count)
    if qc is None:
        return None, 0, 0.0, 0

    t0 = time.perf_counter()
    compiled = transpile(qc, _sim, optimization_level=1)
    job = _sim.run(compiled, shots=shots)
    counts = job.result().get_counts()
    elapsed = (time.perf_counter() - t0) * 1000
    depth = compiled.depth()

    total_states = 2 ** n_count
    top = max(counts, key=counts.get)

    # Examine the strongest peaks; recover the smallest valid order r
    best_r = None
    for outcome in sorted(counts, key=counts.get, reverse=True)[:6]:
        measured = int(outcome, 2)
        if measured == 0:
            continue
        phase = measured / total_states
        r = Fraction(phase).limit_denominator(N).denominator
        if r > 0 and pow(a, r, N) == 1:
            if best_r is None or r < best_r:
                best_r = r
    return best_r, counts[top], elapsed, depth


def sim_shors_dh(p, g, A, seed):
    """
    Use Shor's order-finding to recover DH private key.
    Finds r = order of g mod p, then solves g^x ≡ A (mod p).
    Only feasible for small demonstration primes.
    """
    random.seed(seed)
    # Counting qubits: ~2*ceil(log2 N) gives enough phase resolution to
    # resolve multiples of 1/r reliably (Nielsen & Chuang, 2010). Capped at 8
    # to keep the statevector simulation tractable on a laptop.
    n_count = min(max(4, 2 * math.ceil(math.log2(p))), 8)
    n_work  = max(math.ceil(math.log2(p + 1)), 1)

    t0 = time.perf_counter()
    # Step 1: Find order r of g mod p using the QPE order-finding circuit
    order, top_count, qft_time, circuit_depth = run_shors_order_finding(
        g, p, n_count=n_count, shots=512)

    # Step 2: If order found, use it to solve the discrete log g^x ≡ A (mod p)
    recovered = None
    if order and order > 0:
        for x in range(order):
            if pow(g, x, p) == A:
                recovered = x
                break

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        'p': p, 'g': g,
        'order_found': order,
        'recovered_key': recovered,
        'success': recovered is not None,
        'n_count_qubits': n_count,
        'n_total_qubits': n_count + n_work,
        'circuit_depth': circuit_depth,
        'qft_time_ms': round(qft_time, 3),
        'total_time_ms': round(elapsed, 3),
    }


# ── Experiment 9 — run Shor's order-finding across small moduli ──
# Order-finding requires only gcd(g, N) = 1 (not primality). The set below
# mixes prime moduli (7, 11, 13, 17 — relevant to the discrete-log / DH
# attack) and composite moduli (15, 21 — relevant to the factoring attack),
# demonstrating the shared quantum order-finding subroutine on both.
SHORS_MODULI = [7, 11, 13, 15, 17, 21]  # small N for QPE feasibility
exp9_rows = []
shors_success_rates = []
shors_times = []
shors_orders = []
shors_depths = []
shors_qubits = []

if QISKIT_OK:
    for p in SHORS_MODULI:
        g = 2
        # Fixed private key a=2 and public key A = g^a mod p.
        # a=2 is chosen (rather than 3) because for N=7 the order of g=2 is
        # exactly 3, so a=3 would give A = 2^3 mod 7 = 1, making the discrete
        # log ambiguous (x=0 and x=3 both satisfy g^x=1). a=2 yields a unique
        # smallest solution x=2 for every prime in the set, so the recovered
        # key can be unambiguously verified against the known private key.
        a_priv = 2
        A_pub  = pow(g, a_priv, p)
        successes, times, orders = [], [], []
        depth_p, qubits_p = 0, 0

        for s in range(10):  # 10 seeds — QPE measurement is probabilistic
            random.seed(s)
            r = sim_shors_dh(p, g, A_pub, s)
            successes.append(1 if r['success'] else 0)
            times.append(r['total_time_ms'])
            depth_p, qubits_p = r['circuit_depth'], r['n_total_qubits']
            if r['order_found']:
                orders.append(r['order_found'])
            exp9_rows.append({
                'experiment': 'Shors_DH',
                'modulus_N': p, 'generator_g': g,
                'private_key_a': a_priv, 'public_key_A': A_pub,
                'seed': s,
                'order_found': r['order_found'],
                'recovered_key': r['recovered_key'],
                'success': r['success'],
                'n_count_qubits': r['n_count_qubits'],
                'n_total_qubits': r['n_total_qubits'],
                'circuit_depth': r['circuit_depth'],
                'qft_time_ms': r['qft_time_ms'],
                'total_time_ms': round(r['total_time_ms'], 3),
            })

        sr   = np.mean(successes) * 100
        tmean = np.mean(times)
        omean = np.mean(orders) if orders else 0
        shors_success_rates.append(sr)
        shors_times.append(tmean)
        shors_orders.append(omean)
        shors_depths.append(depth_p)
        shors_qubits.append(qubits_p)
        print(f"  Shor's N={p:3d}  success={sr:.0f}%  order={omean:.1f}  "
              f"qubits={qubits_p}  depth={depth_p}  time={tmean:.1f}ms")

    save_csv('exp9_shors_algorithm.csv', exp9_rows)

    # ── Experiment 9 graph ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Experiment 9 — Shor's Algorithm: Quantum Order Finding on DH\n"
        "(Qiskit AerSimulator — Quantum Phase Estimation + Inverse QFT, 10 seeds per N)",
        fontsize=12
    )

    # Panel 1 — Success rate vs N
    bar_col = ['#43A047' if s == 100 else '#FFA726' if s >= 50 else '#EF5350'
               for s in shors_success_rates]
    bars = axes[0].bar([str(p) for p in SHORS_MODULI], shors_success_rates,
                       color=bar_col, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Modulus N'); axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title("Shor's Order Finding — Success Rate vs N")
    axes[0].set_ylim(0, 115)
    for bar, val in zip(bars, shors_success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2 — Circuit depth vs N (real transpiled depth)
    axes[1].bar([str(p) for p in SHORS_MODULI], shors_depths,
                color='#7E57C2', edgecolor='white', linewidth=0.5)
    axes[1].set_xlabel('Modulus N'); axes[1].set_ylabel('Transpiled Circuit Depth (gates)')
    axes[1].set_title("Quantum Circuit Depth vs Problem Size N")
    # annotate total qubit count
    for i, (p, q) in enumerate(zip(SHORS_MODULI, shors_qubits)):
        axes[1].text(i, shors_depths[i] + max(shors_depths) * 0.01,
                     f'{q}q', ha='center', fontsize=9, color='#4527A0')

    # Panel 3 — Execution time vs N
    axes[2].bar([str(p) for p in SHORS_MODULI], shors_times,
                color='#0288D1', edgecolor='white', linewidth=0.5)
    axes[2].set_xlabel('Modulus N'); axes[2].set_ylabel('Mean Execution Time (ms)')
    axes[2].set_title("Quantum Circuit Execution Time vs N")

    # Add annotation box explaining what Shor's does
    textstr = (
        "Shor's Algorithm (Quantum)\n"
        "─────────────────────────\n"
        "Step 1: Apply H gates → superposition\n"
        "Step 2: Controlled-U gates (mod exp)\n"
        "Step 3: Inverse QFT on phase register\n"
        "Step 4: Measure → extract order r\n"
        "Step 5: Use r to solve discrete log\n"
        "─────────────────────────\n"
        f"Classical BSGS: O(√p) steps\n"
        f"Shor's (quantum): O((log N)³)"
    )
    axes[2].text(1.08, 0.97, textstr, transform=axes[2].transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9),
                 fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('graphs/exp9_shors_algorithm.png', bbox_inches='tight')
    plt.close()
    print("  Graph saved: graphs/exp9_shors_algorithm.png")

    # ── Bonus: Shor's vs BSGS complexity comparison ──────────
    fig, ax = plt.subplots(figsize=(10, 6))
    Ns = list(range(5, 400))
    bsgs_steps  = [math.sqrt(n) for n in Ns]
    shors_steps = [(math.log2(n))**3 for n in Ns]

    ax.plot(Ns, bsgs_steps,  color=COLORS['dh'],  linewidth=2.5, label='Classical BSGS — O(√N)')
    ax.plot(Ns, shors_steps, color='#7E57C2', linewidth=2.5, linestyle='--',
            label="Shor's Algorithm — O((log N)³)")
    ax.axvline(63, color='gray', linestyle=':', alpha=0.7, label='Simulation feasibility limit (N=63)')
    ax.fill_betweenx([0, max(bsgs_steps)], 0, 63, alpha=0.05, color='green',
                     label='Qiskit simulation range')

    # Mark actual test points
    for p in SHORS_MODULI:
        ax.scatter(p, math.sqrt(p), color=COLORS['dh'], s=80, zorder=5)
        ax.scatter(p, (math.log2(p))**3, color='#7E57C2', s=80, zorder=5)

    ax.set_xlabel('Problem Size N (prime modulus)')
    ax.set_ylabel('Approximate Computation Steps')
    ax.set_title("Shor's Algorithm vs Classical BSGS: Attack Complexity Comparison\n"
                 "(Shor's scales polynomially — BSGS scales as square root of N)")
    ax.legend(fontsize=9); ax.set_ylim(0, max(bsgs_steps) * 1.1)
    plt.tight_layout()
    plt.savefig('graphs/exp9_shors_vs_bsgs_complexity.png', bbox_inches='tight')
    plt.close()
    print("  Graph saved: graphs/exp9_shors_vs_bsgs_complexity.png")

else:
    print("  Qiskit not available — skipping Experiment 9 (Shor's algorithm)")
    print("  Install with: pip install qiskit qiskit-aer")


# ══════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  All experiments complete!")
print("=" * 60)
print("\n  CSV files saved in:  results/")
print("  Graphs saved in:     graphs/")
print("\n  Files generated:")
for f in sorted(os.listdir('results')): print(f"    results/{f}")
for f in sorted(os.listdir('graphs')):  print(f"    graphs/{f}")
print("\n  Install requirements (if not already done):")
print("    pip install qiskit qiskit-aer numpy matplotlib cryptography")
print()
