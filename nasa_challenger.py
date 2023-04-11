# UNIVERSIDADE DE SÃO PAULO
# INTRODUÇÃO AO PYTHON E MACHINE LEARNING
# GLM - MODELOS LOGÍSTICOS BINÁRIOS E MULTINOMIAIS
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários

import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
import statsmodels.formula.api as smf # estimação do modelo logístico binário

import warnings
warnings.filterwarnings('ignore')

# In[ ]:
#############################################################################
#           REGRESSÃO LOGÍSTICA BINÁRIA E PROCEDIMENTO STEPWISE             #        
#               EXEMPLO 02 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_challenger = pd.read_csv('challenger.csv',delimiter=',')
df_challenger

#Características das variáveis do dataset
df_challenger.info()

#Estatísticas univariadas
df_challenger.describe()

#desgaste: quantidade de vezes em que ocorreu stress térmico
#temperatura: temperatura de lançamento (graus ºF)
#pressão: pressão de verificação de vazamento (psi: libra-força por
    #polegada ao quadrado)
#t: teste para o lançamento (id)


# In[ ]: Estimação de um modelo logístico binário

#Não há uma variável binária para servir como uma variável dependente, certo?
#Então vamos criá-la, considerando a ocorrência de desgastes de peças como a
#ocorrência de um evento que chamaremos de 'falha':

df_challenger.loc[df_challenger['desgaste'] != 0 , 'falha'] = 1
df_challenger.loc[df_challenger['desgaste'] == 0, 'falha'] = 0

df_challenger


# In[ ]: Estimação do modelo logístico binário

modelo_challenger = sm.Logit.from_formula('falha ~ temperatura + pressão',
                                          df_challenger).fit()

#Parâmetros do modelo
modelo_challenger.summary()


# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.process import stepwise

#Estimação do modelo por meio do procedimento Stepwise
step_challenger = stepwise(modelo_challenger, pvalue_limit=0.05)


# In[ ]: Fazendo predições para o modelo 'step_challenger'

#Apenas como curiosidade, vamos criar uma função que calcula a temperatura
#em graus Celsius a partir da temperatura em graus Fahrenheit:

def celsius(far):
    celsius = 5*((far-32)/9)
    print(celsius)

celsius(70)
celsius(77)
celsius(34) #temperatura no momento do lançamento

#Exemplo 1: qual a probabilidade média de falha a 70ºF (~21.11ºC)?
step_challenger.predict(pd.DataFrame({'temperatura':[70]}))

#Exemplo 2: qual a probabilidade média de falha a 77ºF (25ºC)?
step_challenger.predict(pd.DataFrame({'temperatura':[77]}))

#Exemplo 3: qual a probabilidade média de falha a 34ºF (~1.11ºC)?
#temperatura no momento do lançamento
step_challenger.predict(pd.DataFrame({'temperatura':[34]}))


# In[ ]: Atribuindo uma coluna no dataframe para os resultados

df_challenger['phat'] = step_challenger.predict()


# In[ ]: Construção da sigmoide
#Probabilidade de evento em função da variável 'temperatura'    

plt.figure(figsize=(15,10))
sns.regplot(x = df_challenger.temperatura, y = df_challenger.falha,
            data=df_challenger, logistic=True, ci=None, color='indigo',
            marker='o', scatter_kws={'color':'indigo', 'alpha':0.5, 's':170})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Temperatura em ºF', fontsize=17)
plt.ylabel('Probabilidade de Falha', fontsize=17)
plt.show


# In[ ]: Nossa homenagem aos astronautas

from PIL import Image
import requests
from io import BytesIO

url = "https://img.ibxk.com.br///2016/01/29/29182307148581.jpg?w=1200&h=675&mode=crop&scale=both"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.show()    
