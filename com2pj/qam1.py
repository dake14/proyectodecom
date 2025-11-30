import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from funciones.canal import canal_kumar_variable   # tu canal

# ===================== 1. PARÁMETROS DEL SISTEMA =====================

num_bits        = 2000          # número de bits
samples_per_sym = 30            # sps
Rb              = 1000          # tasa de bits [bits/s]
fs              = Rb * samples_per_sym

carrier_freq    = 5000          # Hz (para visualización en pasabanda)
rolloff         = 0.25          # roll-off RRC
rrc_span_syms   = 30            # span en símbolos

M               = 32             # 4, 16, 32 ...
snr_db          = 10           # SNR en dB
tipo_fading     = 'Rician'      # 'nulo', 'Rayleigh', 'Rician'
nivel_isi       = 'alto'        # 'nulo', 'bajo', 'medio', 'alto'

np.random.seed(0)

# ===================== 2. MODULACIÓN QAM: bits <-> símbolos =====================

def bits_to_mqam_symbols(bits, M):
    """
    Bits {0,1} -> símbolos M-QAM rectangulares normalizados (E_s ≈ 1).
    Soporta M = 4,16,32,... tal que M = 2^k.
    Devuelve:
      symbols : símbolos complejos
      num_pad : bits de padding añadidos
      scale   : factor de escala (E_s ≈ 1)
      all_I   : niveles PAM en I (sin escalar)
      all_Q   : niveles PAM en Q (sin escalar)
    """
    k = int(np.log2(M))          # bits por símbolo
    k_I = k // 2                 # bits para eje I
    k_Q = k - k_I                # bits para eje Q

    # Padding para múltiplo de k
    num_pad = (-len(bits)) % k
    if num_pad > 0:
        bits = np.hstack([bits, np.zeros(num_pad, dtype=int)])

    bits_2d = bits.reshape(-1, k)
    bits_I = bits_2d[:, :k_I]
    bits_Q = bits_2d[:, k_I:]

    # Índices de nivel PAM en cada eje
    idx_I = bits_I.dot(1 << np.arange(k_I-1, -1, -1)) if k_I > 0 else np.zeros(len(bits_2d), int)
    idx_Q = bits_Q.dot(1 << np.arange(k_Q-1, -1, -1)) if k_Q > 0 else np.zeros(len(bits_2d), int)

    # Niveles PAM: -(L-1), ..., -1, 1, ..., (L-1)
    L_I = 2**k_I if k_I > 0 else 1
    L_Q = 2**k_Q if k_Q > 0 else 1

    all_I = 2*np.arange(L_I) - (L_I - 1)  # [-3,-1,1,3] para 16-QAM
    all_Q = 2*np.arange(L_Q) - (L_Q - 1)  # etc.

    levels_I = all_I[idx_I]
    levels_Q = all_Q[idx_Q]

    # Normalización de energía promedio
    II, QQ = np.meshgrid(all_I, all_Q)
    Es_avg = np.mean(II**2 + QQ**2)
    scale = 1 / np.sqrt(Es_avg)

    symbols = scale * (levels_I + 1j*levels_Q)
    return symbols, num_pad, scale, all_I, all_Q


def mqam_symbols_to_bits(symbols_rx, M, scale, all_I, all_Q):
    """
    Símbolos QAM recibidos -> bits {0,1} (detección dura).
    Usa la misma rejilla rectangular que bits_to_mqam_symbols.
    """
    k = int(np.log2(M))
    k_I = k // 2
    k_Q = k - k_I

    # Quitamos escala para trabajar en niveles enteros
    re = np.real(symbols_rx) / scale
    im = np.imag(symbols_rx) / scale

    I_levels = all_I
    Q_levels = all_Q

    I_idx_hat = np.argmin(np.abs(re[:, None] - I_levels[None, :]), axis=1)
    Q_idx_hat = np.argmin(np.abs(im[:, None] - Q_levels[None, :]), axis=1)

    bits_I = (((I_idx_hat[:, None]).astype(int)
               >> np.arange(k_I-1, -1, -1)) & 1).astype(int) if k_I > 0 else np.zeros((len(symbols_rx),0), int)
    bits_Q = (((Q_idx_hat[:, None]).astype(int)
               >> np.arange(k_Q-1, -1, -1)) & 1).astype(int) if k_Q > 0 else np.zeros((len(symbols_rx),0), int)

    bits_mat = np.hstack([bits_I, bits_Q])
    return bits_mat.reshape(-1)

# ===================== 3. BITS Y SÍMBOLOS QAM =====================

bits_tx = np.random.randint(0, 2, num_bits)
symbols_tx, num_pad, scale_qam, all_I, all_Q = bits_to_mqam_symbols(bits_tx, M)
num_symbols = len(symbols_tx)

# ===================== 4. FILTRO RRC =====================

def rrc_filter(rolloff, sps, span_syms):
    N = span_syms * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - rolloff + (4*rolloff/np.pi)
        elif np.isclose(abs(ti), 1/(4*rolloff)):
            h[i] = (rolloff/np.sqrt(2)) * (
                (1 + 2/np.pi)*np.sin(np.pi/(4*rolloff)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*rolloff))
            )
        else:
            num = (np.sin(np.pi*ti*(1-rolloff)) +
                   4*rolloff*ti*np.cos(np.pi*ti*(1+rolloff)))
            den = (np.pi*ti*(1-(4*rolloff*ti)**2))
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2))
    return h

rrc_taps = rrc_filter(rolloff, samples_per_sym, rrc_span_syms)

# ===================== 5. TX EN BANDA BASE =====================

tx_upsampled = np.zeros(num_symbols * samples_per_sym, dtype=complex)
tx_upsampled[::samples_per_sym] = symbols_tx

tx_bb = convolve(tx_upsampled, rrc_taps, mode='full')

# ===================== 6. CANAL EN BANDA BASE =====================

rx_bb_channel, h_chan = canal_kumar_variable(
    tx_signal=tx_bb,
    tipo_fading=tipo_fading,
    nivel_isi=nivel_isi,
    snr_db=snr_db,
    max_fase=np.pi/8
)

# ===================== 7. FILTRO CASADO (RRC RX) =====================

rx_bb_matched = convolve(rx_bb_channel, rrc_taps, mode='full')

# ===================== 8. ALINEACIÓN + ECUALIZACIÓN 1-TAP =====================

rrc_delay   = (len(rrc_taps) - 1) // 2
total_delay = 2 * rrc_delay

# Aquí ya estamos alineando: índice 0 ≈ primer símbolo
rx_bb_sync = rx_bb_matched[total_delay : total_delay + len(tx_upsampled)]
tx_bb_sync = tx_bb[rrc_delay     : rrc_delay     + len(tx_upsampled)]

# Muestras símbolo a símbolo (salida del filtro casado)
raw_symbol_samples = rx_bb_sync[::samples_per_sym][:num_symbols]

# Estimación de ganancia compleja y ecualización 1-tap
num = np.vdot(symbols_tx, raw_symbol_samples)
den = np.vdot(symbols_tx, symbols_tx) + 1e-12
a_hat = num / den
rx_sym_aligned = raw_symbol_samples / a_hat

# ===================== 9. DEMODULACIÓN QAM Y BER =====================

bits_rx = mqam_symbols_to_bits(rx_sym_aligned, M, scale_qam, all_I, all_Q)
bits_rx = bits_rx[:len(bits_tx)]

num_errors = np.sum(bits_rx != bits_tx)
ber = num_errors / len(bits_tx)

print(f"\nM-QAM = {M} | Canal: {tipo_fading} | ISI: {nivel_isi} | SNR = {snr_db} dB")
print(f"BER = {ber:.3e}   Errores = {num_errors}/{len(bits_tx)}\n")

# ===================== 10. FORMAS DE ONDA DESDE EL 1er SÍMBOLO =====================

# t_sync = 0 corresponde al PRIMER símbolo (ya quitamos los retardos)
t_sync = np.arange(len(tx_bb_sync)) / fs

tx_bb_plot = tx_bb_sync
rx_bb_plot = rx_bb_sync

tx_passband = np.real(tx_bb_plot * np.exp(1j * 2*np.pi*carrier_freq * t_sync))
rx_passband = np.real(rx_bb_plot * np.exp(1j * 2*np.pi*carrier_freq * t_sync))

def compute_spectrum(x, fs, Nfft=4096):
    N = min(Nfft, len(x))
    Xf = np.fft.fftshift(np.fft.fft(x[:N]))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    mag = np.abs(Xf)
    mag_db = 20 * np.log10(mag / (np.max(mag) + 1e-12))
    return freqs, mag_db

freqs_tx, mag_tx_db = compute_spectrum(tx_passband, fs)
freqs_rx, mag_rx_db = compute_spectrum(rx_passband, fs)

# ===================== 11. GRÁFICAS =====================

# (a) Constelación Rx vs ideal
plt.figure(figsize=(5,5))
plt.scatter(rx_sym_aligned.real, rx_sym_aligned.imag, alpha=0.5, label='Rx')

# Puntos ideales QAM (rejilla)
ideal_I, ideal_Q = np.meshgrid(all_I*scale_qam, all_Q*scale_qam)
ideal_points = (ideal_I + 1j*ideal_Q).ravel()
plt.scatter(ideal_points.real, ideal_points.imag,
            color='red', marker='x', s=80, label='Ideal')

plt.axhline(0,color='k',linewidth=0.5)
plt.axvline(0,color='k',linewidth=0.5)
plt.title(f"Constelación {M}-QAM")
plt.xlabel("I"); plt.ylabel("Q")
plt.grid(True); plt.axis('equal'); plt.legend()
plt.tight_layout()

# (b) Banda base (real) desde el primer símbolo
plt.figure(figsize=(10,4))
Nbb_plot = min(800, len(t_sync))
plt.plot(t_sync[:Nbb_plot],
         np.real(tx_bb_plot[:Nbb_plot]) / np.max(np.abs(tx_bb_plot)),
         label='Tx BB')
plt.plot(t_sync[:Nbb_plot],
         np.real(rx_bb_plot[:Nbb_plot]) / np.max(np.abs(rx_bb_plot)),
         '--', label='Rx BB (filtrada)', alpha=0.8)
plt.title("Señal en banda base (I) - desde el primer símbolo")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud norm.")
plt.grid(True); plt.legend(); plt.tight_layout()

# (c) Pasabanda Tx vs Rx desde el primer símbolo

plt.figure(figsize=(10,4))
Npb_plot = min(800, len(t_sync))

# Normalizamos cada una por su propio máximo para comparar la forma
tx_pb_plot = tx_passband[:Npb_plot] / (np.max(np.abs(tx_passband)) + 1e-12)
rx_pb_plot = rx_passband[:Npb_plot] / (np.max(np.abs(rx_passband)) + 1e-12)

plt.plot(t_sync[:Npb_plot], tx_pb_plot, label='Tx pasa-banda')
plt.plot(t_sync[:Npb_plot], rx_pb_plot, label='Rx pasa-banda', alpha=0.7)

plt.title(f"Señal pasa-banda Tx/Rx - {M}-QAM (desde el primer símbolo)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.legend()
plt.tight_layout()


# (d) Espectro pasa-banda Tx vs Rx
plt.figure(figsize=(8,4))
plt.plot(freqs_tx/1e3, mag_tx_db, label='Tx')
plt.plot(freqs_rx/1e3, mag_rx_db, label='Rx', alpha=0.7)
plt.title("Espectro pasa-banda (Tx antes de canal/ruido, Rx después)")
plt.xlabel("Frecuencia [kHz]"); plt.ylabel("Magnitud [dB]")
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()