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

M               = 32            # 4, 16 -> cuadrada ; 32 -> cross-QAM
snr_db          = 1000           # SNR en dB
tipo_fading     = 'Rician'      # 'nulo', 'Rayleigh', 'Rician'
nivel_isi       = 'medio'        # 'nulo', 'bajo', 'medio', 'alto'

np.random.seed(0)

# ===================== 2. MODULACIÓN QAM: bits <-> símbolos =====================

def build_constellation(M):
    """
    Devuelve los puntos de constelación QAM complejos y normalizados a
    energía promedio ≈ 1.

    - Para M=4,16 -> QAM cuadrada
    - Para M=32   -> QAM cruzada (cross-32QAM)
    """
    if M == 32:
        # 32-QAM cruzada (sin esquinas ±5±5)
        const_raw = np.array([
            -1+1j, -1-1j,  1+1j,  1-1j,
            -1+3j, -1-3j,  1+3j,  1-3j,
            -3+3j, -3-3j,  3+3j,  3-3j,
            -3+1j, -3-1j,  3+1j,  3-1j,
            -5-1j, -5-3j, -5+1j, -5+3j,
             5+1j,  5+3j,  5-1j,  5-3j,
            -1+5j, -3+5j, -1-5j, -3-5j,
             1-5j,  3-5j,  1+5j,  3+5j
        ], dtype=complex)
    else:
        # QAM cuadrada estándar (M=4,16,... que sean cuadrados perfectos)
        sqrt_M = int(np.sqrt(M))
        if sqrt_M * sqrt_M != M:
            raise ValueError("Solo implementado para M cuadrado (4,16) o M=32 cross.")
        levels = 2*np.arange(sqrt_M) - (sqrt_M - 1)   # p.ej. [-3,-1,1,3]
        I, Q = np.meshgrid(levels, levels)
        const_raw = (I + 1j*Q).ravel()

    # Normalizar energía promedio a 1
    Es_avg = np.mean(np.abs(const_raw)**2)
    const_norm = const_raw / np.sqrt(Es_avg)

    return const_norm


def bits_to_mqam_symbols(bits, M):
    """
    Bits {0,1} -> símbolos MQAM usando la constelación (cuadrada o cruzada).
    """
    k = int(np.log2(M))
    # Padding para múltiplo de k
    num_pad = (-len(bits)) % k
    if num_pad > 0:
        bits = np.hstack([bits, np.zeros(num_pad, dtype=int)])

    bits_2d = bits.reshape(-1, k)
    # índice entero 0..M-1 (binario natural)
    idx = bits_2d.dot(1 << np.arange(k-1, -1, -1))

    constellation = build_constellation(M)
    symbols = constellation[idx]

    return symbols, num_pad, constellation


def mqam_symbols_to_bits(symbols_rx, M, constellation):
    """
    Demodulación dura MQAM por vecino más cercano usando la constelación dada.
    """
    k = int(np.log2(M))
    # Distancia al cuadrado a cada punto de constelación
    diff = symbols_rx.reshape(-1, 1) - constellation.reshape(1, -1)
    d2 = np.abs(diff)**2
    idx_hat = np.argmin(d2, axis=1)   # índice del punto más cercano

    # Volver de índice -> bits
    bits_mat = (((idx_hat[:, None]).astype(int)
                 >> np.arange(k-1, -1, -1)) & 1).astype(int)
    return bits_mat.reshape(-1)

# ===================== 3. BITS Y SÍMBOLOS QAM =====================

bits_tx = np.random.randint(0, 2, num_bits)
symbols_tx, num_pad, constellation = bits_to_mqam_symbols(bits_tx, M)
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

# Aquí ya índice 0 ≈ primer símbolo
rx_bb_sync = rx_bb_matched[total_delay : total_delay + len(tx_upsampled)]
tx_bb_sync = tx_bb[rrc_delay     : rrc_delay     + len(tx_upsampled)]

# Muestras símbolo a símbolo (salida del filtro casado)
raw_symbol_samples = rx_bb_sync[::samples_per_sym][:num_symbols]

# Estimación de ganancia compleja y ecualización 1-tap:
# y_k ≈ a * s_k  →  a_hat = (s^H y)/(s^H s)
num = np.vdot(symbols_tx, raw_symbol_samples)
den = np.vdot(symbols_tx, symbols_tx) + 1e-12
a_hat = num / den
rx_sym_aligned = raw_symbol_samples / a_hat

# ===================== 9. DEMODULACIÓN QAM Y BER =====================

bits_rx = mqam_symbols_to_bits(rx_sym_aligned, M, constellation)
bits_rx = bits_rx[:len(bits_tx)]   # quitar padding

num_errors = np.sum(bits_rx != bits_tx)
ber = num_errors / len(bits_tx)

print(f"\nM-QAM = {M} (cross-QAM si M=32) | Canal: {tipo_fading} | ISI: {nivel_isi} | SNR = {snr_db} dB")
print(f"BER = {ber:.3e}   Errores = {num_errors}/{len(bits_tx)}\n")

# ===================== 10. FORMAS DE ONDA DESDE EL 1er SÍMBOLO =====================

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

plt.scatter(constellation.real, constellation.imag,
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
         np.real(tx_bb_plot[:Nbb_plot]) / (np.max(np.abs(tx_bb_plot)) + 1e-12),
         label='Tx BB')
plt.plot(t_sync[:Nbb_plot],
         np.real(rx_bb_plot[:Nbb_plot]) / (np.max(np.abs(rx_bb_plot)) + 1e-12),
         '--', label='Rx BB (filtrada)', alpha=0.8)
plt.title("Señal en banda base (I) - desde el primer símbolo")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud norm.")
plt.grid(True); plt.legend(); plt.tight_layout()

# (c) Pasabanda Tx vs Rx desde el primer símbolo (normalizada)
plt.figure(figsize=(10,4))
Npb_plot = min(800, len(t_sync))

tx_pb_plot = tx_passband[:Npb_plot] / (np.max(np.abs(tx_passband)) + 1e-12)
rx_pb_plot = rx_passband[:Npb_plot] / (np.max(np.abs(rx_passband)) + 1e-12)

plt.plot(t_sync[:Npb_plot], tx_pb_plot, label='Tx pasa-banda')
plt.plot(t_sync[:Npb_plot], rx_pb_plot, label='Rx pasa-banda', alpha=0.7)
plt.title(f"Señal pasa-banda Tx/Rx - {M}-QAM (desde el primer símbolo)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud normalizada")
plt.grid(True); plt.legend(); plt.tight_layout()

# (d) Espectro pasa-banda Tx vs Rx
plt.figure(figsize=(8,4))
plt.plot(freqs_tx/1e3, mag_tx_db, label='Tx')
plt.plot(freqs_rx/1e3, mag_rx_db, label='Rx', alpha=0.7)
plt.title("Espectro pasa-banda (Tx antes de canal/ruido, Rx después)")
plt.xlabel("Frecuencia [kHz]"); plt.ylabel("Magnitud [dB]")
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()
