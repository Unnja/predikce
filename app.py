import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from io import BytesIO

# --- Konfigurace a názvy souborů ---
FILE_T = "mly-0-20000-0-11723-T.csv"
FILE_F = "mly-0-20000-0-11723-F.csv"
FILE_SRA = "mly-0-20000-0-11723-SRA.csv"

# Ignorování varování
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# --- Definice Funkcí ---

def nacti_a_filtruj_data_z_cesty(filepath, time_func, md_func, nova_value_col):
    """Načte CSV z cesty (filename) a vyfiltruje potřebné řádky."""
    try:
        df = pd.read_csv(
            filepath, 
            usecols=['YEAR', 'MONTH', 'TIMEFUNCTION', 'MDFUNCTION', 'VALUE']
        )
        df_filtrovany = df[
            (df['TIMEFUNCTION'] == time_func) & 
            (df['MDFUNCTION'] == md_func)
        ].copy()
        df_final = df_filtrovany[['YEAR', 'MONTH', 'VALUE']]
        df_final = df_final.rename(columns={'VALUE': nova_value_col})
        df_final[nova_value_col] = pd.to_numeric(df_final[nova_value_col], errors='coerce')
        return df_final
    except FileNotFoundError:
        st.error(f"Chyba: Soubor nenalezen: `{filepath}`. Ujisti se, že je ve stejném repozitáři jako `app.py`.")
        return None
    except Exception as e:
        st.error(f"Chyba při zpracování souboru `{filepath}`: {e}")
        return None

@st.cache_data
def zpracuj_data_z_githubu():
    """Hlavní funkce pro zpracování dat a trénink modelu."""
    with st.spinner("Načítám a zpracovávám data z GitHub repozitáře..."):
        # 1. Načtení a filtrace
        df_temp = nacti_a_filtruj_data_z_cesty(FILE_T, 'AVG', 'AVG', 't_avg')
        df_wind = nacti_a_filtruj_data_z_cesty(FILE_F, 'AVG', 'AVG', 'wspd_avg')
        df_precip = nacti_a_filtruj_data_z_cesty(FILE_SRA, '07:00', 'SUM', 'prcp_sum')

        if df_temp is None or df_wind is None or df_precip is None:
            return None, None, None, None

        # 2. Spojení a roční agregace
        df_monthly = pd.merge(df_temp, df_wind, on=['YEAR', 'MONTH'], how='outer')
        df_monthly = pd.merge(df_monthly, df_precip, on=['YEAR', 'MONTH'], how='outer')

        monthly_counts = df_monthly.dropna().groupby('YEAR').size().reset_index(name='month_count')
        complete_years = monthly_counts[monthly_counts['month_count'] == 12]['YEAR']
        df_monthly_complete = df_monthly[df_monthly['YEAR'].isin(complete_years)]

        data_yearly = df_monthly_complete.groupby('YEAR').agg(
            tavg=('t_avg', 'mean'),
            wspd=('wspd_avg', 'mean'),
            prcp=('prcp_sum', 'sum')
        ).reset_index().dropna()

        if data_yearly.empty:
            st.error("Po filtraci na kompletní roky nezbyla žádná data.")
            return None, None, None, None

        # 3. Modelování
        variables = ['tavg', 'wspd', 'prcp']
        models = {}
        results = {}
        X = data_yearly['YEAR'].values.reshape(-1, 1)

        for var in variables:
            y = data_yearly[var].values
            model = LinearRegression()
            model.fit(X, y)
            models[var] = model
            results[var] = {'slope': model.coef_[0], 'intercept': model.intercept_}
            data_yearly[f'{var}_trend'] = model.predict(X)
            
        st.success("Data úspěšně načtena a modely natrénovány.")
        return data_yearly, results, models, df_monthly

def create_plot(var, info, data_yearly, df_predictions, results):
    """Vytvoří Matplotlib graf pro danou proměnnou."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Historická data
    ax.scatter(data_yearly['YEAR'], data_yearly[var], label=f'Skutečná roční data ({var})', alpha=0.7)
    
    # Regresní přímka (trend)
    ax.plot(data_yearly['YEAR'], data_yearly[f'{var}_trend'], color='red', linestyle='--', label=f'Lineární trend ({results[var]["slope"]:.4f} {info["unit"]}/rok)')
    
    # Spojnice k predikcím
    last_year_data = data_yearly['YEAR'].max()
    last_val_data = data_yearly.loc[data_yearly['YEAR'] == last_year_data, f'{var}_trend'].values[0]
    
    first_pred_year = df_predictions.index.min()
    first_pred_val = df_predictions.loc[first_pred_year, f'pred_{var}']
    
    ax.plot([last_year_data, first_pred_year], [last_val_data, first_pred_val], color='red', linestyle=':', label='Extrapolace')
    
    # Budoucí predikce
    ax.plot(df_predictions.index, df_predictions[f'pred_{var}'], color='red', marker='o', linestyle=':', markersize=8)
    
    # Popisky
    ax.set_title(f'Historický vývoj a lineární extrapolace - {info["label"]} (Brno)')
    ax.set_xlabel('Rok')
    ax.set_ylabel(f'{info["label"]} ({info["unit"]})')
    ax.legend()
    ax.grid(True)
    
    return fig

def to_excel(df_monthly, data_yearly, df_predictions, results):
    """Vytvoří Excel soubor v paměti."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_monthly.to_excel(writer, sheet_name='Měsíční_Data_Raw_Filtered', index=False)
        data_yearly.to_excel(writer, sheet_name='Agregovaná_Roční_Data', index=False)
        df_predictions.to_excel(writer, sheet_name='Predikce_Scénáře', index=True)
        df_results = pd.DataFrame(results).T
        df_results.to_excel(writer, sheet_name='Parametry_Modelu')
    
    output.seek(0)
    return output.getvalue()

# --- Rozhraní Aplikace Streamlit ---

st.set_page_config(layout="wide", page_title="Prediktor Klimatu Brno")
st.title("Analýza a predikce klimatu - Brno (stanice 11723)")
st.caption("Tento nástroj provádí lineární regresi na historických datech a extrapoluje trendy do budoucnosti. Slouží jako demonstrace metody a jejích omezení.")

# --- Zpracování dat (volá se automaticky) ---
data_yearly, results, models, df_monthly = zpracuj_data_z_githubu()

# --- Hlavní část aplikace (zobrazí se, jen když data OK) ---
if data_yearly is not None:
    
    # --- Postranní panel pro interaktivitu ---
    st.sidebar.header("Parametry modelu")
    st.sidebar.write("Vypočtený sklon trendu (jednotek/rok):")
    st.sidebar.json({
        "tavg_slope": f"{results['tavg']['slope']:.4f} °C/rok",
        "wspd_slope": f"{results['wspd']['slope']:.4f} m/s/rok",
        "prcp_slope": f"{results['prcp']['slope']:.4f} mm/rok"
    })
    
    st.sidebar.header("Horizonty predikce")
    st.sidebar.info("Zvolte roky pro extrapolaci.")
    current_year = datetime.now().year
    
    h1 = st.sidebar.slider("Horizont 1 (roky od teď)", 1, 50, 10)
    h2 = st.sidebar.slider("Horizont 2 (roky od teď)", 51, 500, 100)
    h3 = st.sidebar.slider("Horizont 3 (roky od teď)", 501, 2000, 1000)
    
    horizons_years = [current_year + h1, current_year + h2, current_year + h3]
    
    # --- Dynamická predikce ---
    predictions = {}
    for var, model in models.items():
        future_years = np.array(horizons_years).reshape(-1, 1)
        future_predictions = model.predict(future_years)
        predictions[f'pred_{var}'] = future_predictions

    df_predictions = pd.DataFrame(predictions, index=horizons_years)
    df_predictions.index.name = 'Year'
    df_predictions = df_predictions.round(2)

    st.header("Interaktivní predikce / Scénáře")
    st.dataframe(df_predictions, use_container_width=True)
    st.warning("Pamatujte: Predikce na 100 a 1000 let jsou čistě hypotetická lineární extrapolace a nedávají reálný vědecký smysl. Slouží k demonstraci limitů metody.")

    # --- Definice proměnných pro grafy ---
    variables_to_plot = {
        'tavg': {'unit': '°C', 'label': 'Průměrná teplota'},
        'wspd': {'unit': 'm/s', 'label': 'Průměrná rychlost větru'},
        'prcp': {'unit': 'mm', 'label': 'Celkové roční srážky'}
    }

    # --- Vykreslení grafů v záložkách ---
    tab_t, tab_w, tab_p = st.tabs(["Graf Teplota", "Graf Vítr", "Graf Srážky"])

    with tab_t:
        st.subheader("Vývoj a extrapolace průměrné roční teploty")
        fig_t = create_plot('tavg', variables_to_plot['tavg'], data_yearly, df_predictions, results)
        st.pyplot(fig_t)

    with tab_w:
        st.subheader("Vývoj a extrapolace průměrné roční rychlosti větru")
        fig_w = create_plot('wspd', variables_to_plot['wspd'], data_yearly, df_predictions, results)
        st.pyplot(fig_w)

    with tab_p:
        st.subheader("Vývoj a extrapolace celkových ročních srážek")
        fig_p = create_plot('prcp', variables_to_plot['prcp'], data_yearly, df_predictions, results)
        st.pyplot(fig_p)
        
    # --- Zobrazení dat a download ---
    st.divider()
    st.header("Výstupní data")
    
    # Vytvoření Excelu v paměti
    excel_data = to_excel(df_monthly, data_yearly, df_predictions, results)
    
    st.download_button(
        label="Stáhnout kompletní výstup jako Excel",
        data=excel_data,
        file_name=f"vystup_brno_pocasi_{current_year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    with st.expander("Zobrazit zpracovaná roční data (agregovaná)"):
        st.dataframe(data_yearly, use_container_width=True)
        
    with st.expander("Zobrazit měsíční data (filtrovaná, před agregací)"):
        st.dataframe(df_monthly, use_container_width=True)

else:
    st.info("Čekání na data... Pokud se nic neděje, zkontrolujte chybové hlášky výše.")
