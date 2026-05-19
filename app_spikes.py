import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Averaged Curves", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Interface Language:", ["Français", "English"])

# --- DICTIONNAIRE DE TRADUCTION ---
T = {
    "title": {"Français": "Courbes Moyennées f-I & I-V", "English": "Averaged f-I & I-V Curves"},
    "subtitle": {"Français": "Analyse de Population & Excitabilité | Manzoni Lab", "English": "Population Analysis & Excitability | Manzoni Lab"},
    "upload": {"Français": "📂 1. Charger les fichiers", "English": "📂 1. Upload Files"},
    "upload_help": {"Français": "Sélectionnez plusieurs fichiers terminant par '_Sweeps.csv'", "English": "Select multiple files ending in '_Sweeps.csv'"},
    "settings": {"Français": "⚙️ 2. Paramètres Visuels", "English": "⚙️ 2. Visual Settings"},
    "err_type": {"Français": "Type d'Erreur (Ombrage)", "English": "Error Type (Shading)"},
    "err_sem": {"Français": "SEM (Erreur Standard)", "English": "SEM (Standard Error)"},
    "err_ci": {"Français": "CI (95% Confiance)", "English": "CI (95% Confidence)"},
    "tab_graphs": {"Français": "📈 Visualisation des Courbes", "English": "📈 Curve Visualization"},
    "tab_data": {"Français": "🔢 Données Moyennées (Export)", "English": "🔢 Averaged Data (Export)"},
    "tab_howto": {"Français": "📚 Mode d'Emploi (How-To)", "English": "📚 User Guide (How-To)"},
    "export_btn": {"Français": "💾 Exporter les Moyennes (CSV)", "English": "💾 Export Averages (CSV)"},
    "fi_title": {"Français": "Courbe d'Excitabilité (f-I)", "English": "Excitability Curve (f-I)"},
    "iv_title": {"Français": "Relation Courant-Voltage (I-V)", "English": "Current-Voltage Relation (I-V)"},
}

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Neurosciences") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header(T["upload"][lang])
uploaded_files = st.sidebar.file_uploader(
    T["upload_help"][lang], 
    type=["csv"], 
    accept_multiple_files=True
)

st.sidebar.header(T["settings"][lang])
error_choice = st.sidebar.radio(
    T["err_type"][lang], 
    [T["err_sem"][lang], T["err_ci"][lang]]
)
# Définition du paramètre Seaborn en fonction du choix
err_bar_style = 'se' if error_choice == T["err_sem"][lang] else ('ci', 95)

# --- CORPS DE L'APPLICATION ---
if not uploaded_files:
    st.info("👈 " + ("Veuillez charger vos fichiers '_Sweeps.csv' dans le menu latéral pour commencer." if lang == "Français" else "Please upload your '_Sweeps.csv' files in the sidebar to begin."))
else:
    df_list = []
    cell_count = 0
    
    for f in uploaded_files:
        if "Sweeps" in f.name or "sweeps" in f.name.lower():
            # CORRECTION CRITIQUE : Réinitialise le pointeur du fichier pour les re-runs de Streamlit
            f.seek(0) 
            
            try:
                df = pd.read_csv(f)
                df['Cell_ID'] = f.name.replace("_Sweeps.csv", "").replace(".csv", "")
                df_list.append(df)
                cell_count += 1
            except Exception as e:
                st.warning(f"Impossible de lire le fichier {f.name} : {e}")
            
    if not df_list:
        st.error("Aucun fichier valide trouvé. Assurez-vous qu'ils contiennent 'Sweeps' dans leur nom." if lang == "Français" else "No valid files found. Ensure they contain 'Sweeps' in the filename.")
    else:
        master_df = pd.concat(df_list, ignore_index=True)
        
        required_cols = ['I_inj', 'Nb_Spikes', 'V_steady']
        
        # Vérification des colonnes
        if not all(col in master_df.columns for col in required_cols):
            st.error(f"Format incorrect. Les colonnes requises sont : {required_cols}. Colonnes trouvées : {list(master_df.columns)}")
        else:
            # CORRECTION : S'assure que les données sont numériques (évite les bugs liés aux formats CSV)
            for col in required_cols:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
            
            st.success(f"✅ {cell_count} " + ("cellules fusionnées avec succès." if lang == "Français" else "cells successfully merged."))
            
            # --- ONGLETS ---
            tab1, tab2, tab3 = st.tabs([T["tab_graphs"][lang], T["tab_data"][lang], T["tab_howto"][lang]])

            # --- ONGLET 1 : GRAPHIQUES ---
            with tab1:
                # CORRECTION : Style Seaborn robuste indépendant de la version Matplotlib
                sns.set_style("white")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Sous-échantillonnage pour la courbe f-I
                df_depol = master_df[master_df['I_inj'] >= 0]
                
                # Graphe f-I
                sns.lineplot(
                    data=df_depol, x='I_inj', y='Nb_Spikes', 
                    errorbar=err_bar_style, err_style="band", 
                    marker='o', color='firebrick', ax=ax1, linewidth=2
                )
                ax1.set_title(T["fi_title"][lang], fontweight='bold')
                ax1.set_xlabel("Injected Current (nA/pA)")
                ax1.set_ylabel("Spike Count (Hz)")
                ax1.grid(True, linestyle='--', alpha=0.5)

                # Graphe I-V
                sns.lineplot(
                    data=master_df, x='I_inj', y='V_steady', 
                    errorbar=err_bar_style, err_style="band", 
                    marker='s', color='royalblue', ax=ax2, linewidth=2
                )
                ax2.set_title(T["iv_title"][lang], fontweight='bold')
                ax2.set_xlabel("Injected Current (nA/pA)")
                ax2.set_ylabel("
