#Poster econometría 2#
#Metodología Box Jenkins
# 1. IMPORTAR LIBRERÍAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import jarque_bera, probplot


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#Configuración para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 2. Base de datos
data_path = Path("C:\\Users\\USUARIO\\Downloads\\dato_posterIPC.csv")
df = pd.read_csv(data_path, header=0, names=['Fecha', 'Valor'], parse_dates=[0])
print(df.head())
