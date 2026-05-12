import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Batch Analysis Expert", layout="wide")

st.markdown("# 📊 Analyse de Groupe & Statistiques")
st.markdown("### Consolidation : Propriétés Passives, f-I & I-V")

# --- CHARGEMENT DES DONNÉES ---
uploaded_files = st.file_uploader("📂 Charger vos fichiers CSV (_Global et _Sweeps)", type="csv", accept_multiple_files=True)

if uploaded_files:
    global_dfs = []
    sweeps_dfs = []
    cell_ids = set()

    # 1. Lecture et nettoyage des noms
    for f in uploaded_files:
        df = pd.read_csv(f)
        # Extraction propre du Cell_ID (ex: 2026_04_29_0004X)
        cell_id = f.name.replace("_Global.csv", "").replace("_Sweeps.csv", "").replace(".abf", "")
        cell_ids.add(cell_id)
        
        df['Cell_ID'] = cell_id
        
        if "_Global" in f.name:
            global_dfs.append(df)
        elif "_Sweeps" in f.name:
            sweeps_dfs.append(df)

    if not global_dfs or not sweeps_dfs:
        st.error("⚠️ Erreur : Chargez à la fois les fichiers '_Global.csv' et '_Sweeps.csv'.")
    else:
        df_g = pd.concat(global_dfs, ignore_index=True)
        df_s = pd.concat(sweeps_dfs, ignore_index=True)

        # --- 2. MAPPING DES CONDITIONS (NOUVEAUTÉ) ---
        st.divider()
        st.subheader("📝 Assignation des Conditions Biologiques")
        st.info("💡 **Astuce Manzoni Lab :** Vous pouvez copier une colonne entière depuis Excel (WT, KO...) et la coller directement dans la colonne 'Condition' ci-dessous.")
        
        # Création du tableau de mapping interactif
        mapping_df = pd.DataFrame({
            "Cell_ID": list(cell_ids),
            "Condition": ["WT"] * len(cell_ids) # WT par défaut
        })
        
        # Éditeur interactif
        edited_mapping = st.data_editor(mapping_df, use_container_width=False, hide_index=True)
        
        # Fusion des conditions choisies avec les données
        df_g = df_g.merge(edited_mapping, on="Cell_ID")
        df_s = df_s.merge(edited_mapping, on="Cell_ID")

        st.divider()

        # --- 3. ANALYSE ET GRAPHES ---
        tab1, tab2, tab3 = st.tabs(["💎 Paramètres Intrinsèques", "📈 Courbes f-I / I-V", "📝 Rapport Statistique"])

        with tab1:
            st.subheader("Moyennage des Propriétés Biophysiques")
            
            # Exclusion des colonnes texte pour la table
            numeric_cols = [c for c in df_g.columns if c not in ['Cell_ID', 'File', 'Condition']]
            
            # Tableau récapitulatif
            st.markdown("#### Table : Mean ± SEM par Groupe")
            if not df_g.empty and len(numeric_cols) > 0:
                summary_table = df_g.groupby('Condition')[numeric_cols].agg(['mean', 'sem']).stack(level=0)
                summary_table['Mean ± SEM'] = summary_table.apply(lambda x: f"{x['mean']:.2f} ± {x['sem']:.2f}", axis=1)
                st.dataframe(summary_table[['Mean ± SEM']].unstack(level=1))

            st.markdown("#### Visualisation (Boxplot)")
            col_sel = st.selectbox("Sélectionner un paramètre", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_g, x='Condition', y=col_sel, palette="vlag", ax=ax)
            sns.stripplot(data=df_g, x='Condition', y=col_sel, color=".3", size=5, ax=ax)
            ax.set_title(f"Comparaison : {col_sel}")
            st.pyplot(fig)

        with tab2:
            st.subheader("Analyse des Courbes de Population")
            error_type = st.radio("Affichage des barres d'erreur :", ["SEM (Standard Error)", "CI (95% Confiance)"], horizontal=True)
            err_bar = 'se' if error_type == "SEM" else 95

            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Fréquence de décharge (f-I)**")
                fig_fi, ax_fi = plt.subplots()
                # Exclusion des steps hyperpolarisants pour la courbe f-I (courant < 0)
                df_s_depol = df_s[df_s['I_inj'] >= 0]
                sns.lineplot(data=df_s_depol, x='I_inj', y='Nb_Spikes', hue='Condition', 
                             marker='o', errorbar=err_bar, ax=ax_fi)
                ax_fi.set_ylabel("Nombre de PA (Hz)")
                ax_fi.set_xlabel("Courant (nA / pA)")
                st.pyplot(fig_fi)

            with c2:
                st.markdown("**Relation Courant-Voltage (I-V)**")
                fig_iv, ax_iv = plt.subplots()
                sns.lineplot(data=df_s, x='I_inj', y='V_steady', hue='Condition', 
                             marker='s', errorbar=err_bar, ax=ax_iv)
                ax_iv.set_ylabel("Voltage (mV)")
                ax_iv.set_xlabel("Courant (nA / pA)")
                st.pyplot(fig_iv)

        with tab3:
            st.subheader("Tests Statistiques (Format Publication)")
            
            groups = df_g['Condition'].unique()
            if len(groups) == 2:
                st.markdown(f"#### 1. {col_sel} ({groups[0]} vs {groups[1]})")
                g1 = df_g[df_g['Condition'] == groups[0]][col_sel].dropna()
                g2 = df_g[df_g['Condition'] == groups[1]][col_sel].dropna()
                
                # Normalité (Shapiro) pour choisir entre T-test et Mann-Whitney
                norm1, norm2 = stats.shapiro(g1)[1], stats.shapiro(g2)[1]
                if norm1 > 0.05 and norm2 > 0.05:
                    test_stat, p_val = stats.ttest_ind(g1, g2)
                    st.write(f"Test Paramétrique (Student t-test) : p = **{p_val:.4f}**")
                else:
                    test_stat, p_val = stats.mannwhitneyu(g1, g2)
                    st.write(f"Test Non-Paramétrique (Mann-Whitney) : p = **{p_val:.4f}**")
            
            st.divider()

            st.markdown("#### 2. Courbe f-I (ANOVA à Mesures Répétées)")
            try:
                # Filtrer pour n'avoir que les valeurs >= 0 (dépolarisation)
                df_anova = df_s[df_s['I_inj'] >= 0].copy()
                
                # ANOVA RM
                res_fi = AnovaRM(data=df_anova, depvar='Nb_Spikes', subject='Cell_ID', within=['I_inj'], aggregate_func='mean').fit()
                st.write("**Effet Intra-Sujet (Courant) :**")
                st.write(res_fi.summary())
                
                # Modèle linéaire pour l'Interaction Groupe * Courant
                model = ols('Nb_Spikes ~ C(Condition) * I_inj', data=df_anova).fit()
                st.write("**Interaction (Condition x Courant) :**")
                st.table(sm.stats.anova_lm(model, typ=2))
            except Exception as e:
                st.warning(f"Note sur l'ANOVA RM : Vérifiez que toutes les cellules ont le même protocole d'injection. ({str(e)})")

        # --- EXPORT ---
        st.divider()
        st.subheader("📥 Exportation Master")
        
        col_ex1, col_ex2 = st.columns(2)
        csv_master_g = df_g.to_csv(index=False).encode('utf-8')
        col_ex1.download_button("💾 Master_Global.csv", csv_master_g, "Master_Global_Consolidated.csv", "text/csv", use_container_width=True)
        
        csv_master_s = df_s.to_csv(index=False).encode('utf-8')
        col_ex2.download_button("💾 Master_Curves.csv", csv_master_s, "Master_Sweeps_Consolidated.csv", "text/csv", use_container_width=True)
