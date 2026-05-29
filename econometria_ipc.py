#Poster econometría 2#
#Metodología Box Jenkins

# 1. IMPORTAR LIBRERÍAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import jarque_bera, probplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===============================

# 2. Base de datos
def cargar_datos():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "datos"
    data_path = DATA_DIR / "dato_posterIPC.csv"
    df = pd.read_csv(data_path, header=0, names=['Fecha', 'Valor'], parse_dates=['Fecha'],)
    df.set_index('Fecha', inplace=True)
    print(df.head())
    return df
# %%==============================
# índice de precios al consumidor- USA

# 3. Gráfico de la serie original
def graficar_serie(df_ipc):
    BASE_DIR = Path(__file__).resolve().parent
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.figure(figsize=(12, 6))
    plt.plot(df_ipc.index, df_ipc['Valor'], label='Índice de Precios al Consumidor en EE.UU (1955-presente) ')
    plt.xlabel('Año')
    plt.ylabel('Variación Mensual del IPC (%)')
    plt.title('Serie Temporal del Índice de Precios al Consumidor - EE.UU')
    ticks = np.arange(1945, df_ipc.index.max().year + 1, 10)
    plt.xticks(ticks=ticks, rotation=45)
    plt.xlim(df_ipc.index.min(), df_ipc.index.max())
    plt.savefig(BASE_DIR / "serie_original.png")  
    plt.close()  


#  Gráficos FAC y FACP de la serie original 
def graficar_fac_facp(serie, sufijo="", archivo="fac_facp.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    BASE_DIR = Path(__file__).resolve().parent
    plot_acf(serie, ax=axes[0], lags=40, title="Función de Autocorrelación (ACF)")
    plot_pacf(serie, ax=axes[1], lags=40, title="Función de Autocorrelación Parcial (PACF)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / archivo)
    plt.close()
    print(f"Guardada: {archivo}")


#  Logaritmo de la serie
def logaritmo_y_graficar(df_ipc):
    # Elimina ceros antes de aplicar logaritmo
    df_sin_ceros = df_ipc[df_ipc['Valor'] > 0]
    ipc_serie_log = np.log(df_sin_ceros['Valor'])
    
    # Gráfica de la serie logarítmica
    BASE_DIR = Path(__file__).resolve().parent
    plt.figure(figsize=(12, 6))
    plt.plot(ipc_serie_log.index, ipc_serie_log, label='Serie Logarítmica')
    plt.xlabel('Fecha')
    plt.ylabel('Valor (log)')
    plt.title('Serie Temporal Logarítmica')
    plt.legend()
    plt.savefig(BASE_DIR / "serie_logaritmica.png")
    plt.close()
    
    # FAC, FACP de la serie logarítmica
    graficar_fac_facp(ipc_serie_log.dropna().reset_index(drop=True), "Logarítmica", "fac_facp_log.png")
    return ipc_serie_log

# 5. Prueba de estacionariedad (ADF)
def prueba_adf(serie, nombre="Serie"):
    result = adfuller(serie.dropna())
    print(f"\n── ADF: {nombre} ──")
    print(f"Estadístico: {result[0]:.4f}")
    print(f"p-valor:     {result[1]:.4f}")
    print(f"Lags:        {result[2]}")
    print("Valores críticos:")
    for key, val in result[4].items():
        print(f"  {key}: {val:.4f}")

# Prueba de estacionariedad (KPSS)
def prueba_kpss(serie, nombre="Serie"):
    kpss_result = kpss(serie, regression='c', nlags='auto')
    print(f"\n=== Test KPSS: {nombre} ===")
    print("Estadístico KPSS:", kpss_result[0])
    print("p-valor:", kpss_result[1])
    print("Rezagos usados:", kpss_result[2])
    print("Valores críticos:")
    for nivel, valor in kpss_result[3].items():
        print(f"{nivel}: {valor}")


if __name__ == "__main__":
    df_ipc = cargar_datos()
    graficar_serie(df_ipc)
    graficar_fac_facp(df_ipc['Valor'], "Original", "fac_facp_original.png")
    prueba_adf(df_ipc['Valor'], "Original")
    prueba_kpss(df_ipc['Valor'], "Original")
    ipc_serie_log = logaritmo_y_graficar(df_ipc)
    from pathlib import Path
    print(Path(__file__).resolve().parent)
    



