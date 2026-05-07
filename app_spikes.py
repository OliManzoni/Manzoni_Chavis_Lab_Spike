import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Neural Excitability", layout="wide")

# --- EN-TÊTE ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Neural Analysis") 
with col_r:
    st.markdown("# Pipeline Expert : Excitabilité & Morphométrie")
    st.markdown("### Analyse de la Plasticité Synaptique | Standard Nature/Science")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header("📂 1. Chargement")
uploaded_file = st.sidebar.file_uploader("Charger un fichier ABF", type=["abf"])

st.sidebar.header("⚙️ 2. Réglages de Détection")
spike_threshold = st.sidebar.number_input("Seuil de détection (mV)", value=0.0)
dvdt_threshold = st.sidebar.number_input("Seuil dV/dt (mV/ms)", value=15.0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        # DÉTECTION AUTO DES UNITÉS (Via Telegraph)
        unit_i = abf.sweepUnitsC  
        unit_v = abf.sweepUnitsY
        st.sidebar.success(f"Unités détectées : {unit_i} / {unit_v}")
        
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6) 
        
        # Listes d'extraction
        courants, v_stat, v_peak, v_rest_list, n_spikes = [], [], [], [], []
        v_thresh_list, ap_amps, ap_widths, ap_rise, ap_decay, ap_ahp = [], [], [], [], [], []
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_r = np.mean(abf.sweepY[0:idx_start])
            v_s = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_p = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            trace_win = abf.sweepY[idx_start:idx_end]
            peaks, _ = find_peaks(trace_win, height=spike_threshold)
            num_spikes = len(peaks)
            
            vt, amp, width, rise, decay, ahp = [np.nan]*6
            
            if num_spikes > 0:
                pk_idx = peaks[0]
                s_start = max(0, pk_idx - int(sr * 0.015))
                seg = trace_win[s_start:pk_idx]
                
                if len(seg) > 1:
                    smoothed = gaussian_filter1d(seg, sigma=1)
                    dvdt = np.diff(smoothed) / dt_ms
                    cross = np.where(dvdt > dvdt_threshold)[0]
                    if len(cross) > 0:
                        idx_t_seg = cross[0]
                        vt = seg[idx_t_seg]
                        idx_t_glob = s_start + idx_t_seg
                        amp = trace_win[pk_idx] - vt
                        if amp > 0:
                            v50 = vt + 0.5*amp; v10 = vt + 0.1*amp; v90 = vt + 0.9*amp
                            up = trace_win[idx_t_glob:pk_idx]
                            dn = trace_win[pk_idx:min(len(trace_win), pk_idx+int(sr*0.01))]
                            r10 = np.where(up >= v10)[0]; r90 = np.where(up >= v90)[0]
                            if len(r10)>0 and len(r90)>0: rise = (r90[0]-r10[0])*dt_ms
                            d90 = np.where(dn <= v90)[0]; d10 = np.where(dn <= v10)[0]
                            if len(d90)>0 and len(d10)>0: decay = (d10[0]-d90[0])*dt_ms
                            wup = np.where(up >= v50)[0]; wdn = np.where(dn <= v50)[0]
                            if len(wup)>0 and len(wdn)>0: width = ((pk_idx+wdn[0])-(idx_t_glob+wup[0]))*dt_ms
                            ahp = np.min(dn)

            courants.append(i_cmd); v_stat.append(v_s); v_peak.append(v_p)
            v_rest_list.append(v_r); n_spikes.append(num_spikes); v_thresh_list.append(vt)
            ap_amps.append(amp); ap_widths.append(width); ap_rise.append(rise); ap_decay.append(decay); ap_ahp.append(ahp)

        # Calculs Globaux
        v_rest_final = np.mean(v_rest_list)
        neg = [i for i, c in enumerate(courants) if c < 0]
        rin, tau, cm = np.nan, np.nan, np.nan
        if neg:
            neg_s = sorted(neg, key=lambda i: abs(courants[i]))[:4]
            rin = np.polyfit([courants[i] for i in neg_s]+[0], [v_stat[i] for i in neg_s]+[v_rest_final], 1)[0] * (1 if unit_i=="nA" else 1000)
            idx_t = sorted(neg, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_targ = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]) + 0.632*(v_stat[idx_t]-np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]))
            cross_t = np.where(abf.sweepY[idx_start:idx_end] <= v_targ)[0]
            if len(cross_t)>0: 
                tau = (cross_t[0]/sr)*1000.0
                cm = (tau/rin)*1000.0 if rin>0 else np.nan

        # --- AFFICHAGE METRICS ---
        st.subheader("📊 Propriétés Intrinsèques")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vrest", f"{v_rest_final:.1f} mV")
        c2.metric("Rin", f"{rin:.1f} MΩ")
        c3.metric("Cm (Capacitance)", f"{cm:.1f} pF")
        c4.metric("Tau_m", f"{tau:.1f} ms")

        st.divider()

        # --- VISUALISATION INTERACTIVE RESTAURÉE ---
        st.subheader("📈 Visualisations des Traces & Courbes")
        
        # RESTAURATION : Contrôles interactifs divisés en deux colonnes
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            rheo_idx = next((i for i, count in enumerate(n_spikes) if count > 0), 0)
            sw_idx = st.slider("Sélectionner un Sweep individuel", 0, abf.sweepCount-1, int(rheo_idx))
        with col_v2:
            stk_indices = st.multiselect("Sélectionner les sweeps pour l'Overlay", 
                                         list(range(abf.sweepCount)), 
                                         default=[0, abf.sweepCount//2, abf.sweepCount-1])
        
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Trace Individuelle
        ax1 = fig.add_subplot(gs[0, 0])
        abf.setSweep(sw_idx)
        ax1.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
        if not np.isnan(v_thresh_list[sw_idx]):
            ax1.axhline(v_thresh_list[sw_idx], color='red', ls='--', alpha=0.6, label="Seuil")
        ax1.set_title(f"Trace Individuelle : Sweep {sw_idx} ({courants[sw_idx]:.1f} {unit_i})", fontweight='bold')
        ax1.set_ylabel("mV")

        # 2. Overlay (RESTAURÉ AVEC MULTISELECT ET COULEURS)
        ax2 = fig.add_subplot(gs[0, 1])
        cmap = plt.colormaps.get_cmap('viridis')
        for i, s in enumerate(stk_indices):
            abf.setSweep(s)
            ax2.plot(abf.sweepX, abf.sweepY, color=cmap(i/max(1, len(stk_indices))), alpha=0.8, lw=0.8)
        ax2.set_title(f"Superposition de {len(stk_indices)} traces", fontweight='bold')
        ax2.set_ylabel("mV")

        # 3. Courbe I-V
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(courants, v_stat, 'o-', color='tab:blue', label="Steady-state")
        ax3.plot(courants, v_peak, 'x--', color='tab:blue', alpha=0.5, label="Peak (Sag)")
        ax3.axhline(v_rest_final, color='gray', ls=':', lw=1)
        
        # RESTAURATION : Point rouge sur la courbe I-V
        ax3.plot(courants[sw_idx], v_stat[sw_idx], 'ro', markersize=9, zorder=5, label=f"Sweep {sw_idx}")
        
        ax3.set_title("Relation I-V", fontweight='bold')
        ax3.set_xlabel(f"Injection de courant ({unit_i})")
        ax3.set_ylabel("Potentiel membranaire (mV)")
        ax3.legend()

        # 4. Courbe f-I
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(courants, n_spikes, 's-', color='tab:orange')
        
        # RESTAURATION : Point rouge sur la courbe f-I
        ax4.plot(courants[sw_idx], n_spikes[sw_idx], 'ro', markersize=9, zorder=5)
        
        ax4.set_title("Excitabilité (Courbe f-I)", fontweight='bold')
        ax4.set_xlabel(f"Injection de courant ({unit_i})")
        ax4.set_ylabel("Nombre de Potentiels d'Action")
        
        st.pyplot(fig)

        # --- EXPORT ---
        st.divider()
        st.subheader("📥 Exportation des Résultats")
        df_exp = pd.DataFrame({
            "Sweep": abf.sweepList, "I_inj": courants, "Nb_AP_par_step": n_spikes,
            "V_steady": v_stat, "V_threshold": v_thresh_list, "AP_Amp": ap_amps, 
            "AP_Width_ms": ap_widths, "AP_Rise_ms": ap_rise, "AP_Decay_ms": ap_decay, "AP_AHP": ap_ahp
        })
        st.download_button("💾 Télécharger les données morphométriques (CSV)", df_exp.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_results.csv")

        # --- AIDE MÉMOIRE ---
        with st.expander("📖 Aide Mémoire : Formalisme & Méthode de Capacitance"):
            st.markdown("### Modèle RC & Capacitance ($C_m$)")
            st.latex(r"C_m = \frac{\tau_m}{R_{in}}")
            st.markdown("""
            * **Extraction :** $R_{in}$ est calculée sur les pulses faibles pour rester en zone linéaire. $\\tau_m$ est le temps de charge à 63.2%.
            * **Limites (Space-Clamp) :** Dans les neurones complexes, les dendrites distales ne sont pas isopotentielles. Cela induit une erreur systématique où $C_m$ est sous-estimée car l'amplificateur ne voit qu'une fraction de la membrane totale.
            * **Morphométrie PA :** Le seuil est détecté à **15 mV/ms** sur la dérivée lissée. La demi-largeur (*half-width*) est un marqueur de la cinétique des canaux potassiques.
            """)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
