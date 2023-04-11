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
#                      REGRESSÃO LOGÍSTICA MULTINOMIAL                      #
#                EXEMPLO 04 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_atrasado_multinomial = pd.read_csv('atrasado_multinomial.csv',delimiter=',')
df_atrasado_multinomial

#Características das variáveis do dataset
df_atrasado_multinomial.info()

#Estatísticas univariadas
df_atrasado_multinomial.describe()


# In[ ]: Note que a variável Y 'atrasado' está definida como objeto

#Tabela de frequências absolutas da variável 'atrasado' com labels
df_atrasado_multinomial['atrasado'].value_counts(sort=False)

#Criando uma variável 'atrasado2' a partir da variável 'atrasado',
#com labels iguais a 0, 1 e 2 e com tipo 'int' (poderia também ser do tipo
#'float'), a fim de que seja possível estimar o modelo por meio
#da função 'MNLogit'
df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'nao chegou atrasado',
                            'atrasado2'] = 0 #categoria de referência
df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'chegou atrasado primeira aula',
                            'atrasado2'] = 1
df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'chegou atrasado segunda aula',
                            'atrasado2'] = 2

df_atrasado_multinomial['atrasado2'] =\
    df_atrasado_multinomial['atrasado2'].astype('int64')


# In[ ]: Estimação do modelo logístico multinomial

from statsmodels.discrete.discrete_model import MNLogit

x = df_atrasado_multinomial.drop(columns=['estudante','atrasado','atrasado2'])
y = df_atrasado_multinomial['atrasado2']

#Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

#Estimação do modelo - função 'MNLogit' do pacote
#'statsmodels.discrete.discrete_model'
modelo_atrasado = MNLogit(endog=y, exog=X).fit()

#Parâmetros do modelo
modelo_atrasado.summary()


# In[ ]: Vamos definir uma função 'Qui2' para se extrair a estatística geral
# do modelo

def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,1)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df


# In[ ]: Estatística geral do 'modelo_atrasado'

Qui2(modelo_atrasado)


# In[ ]: Fazendo predições para o 'modelo_atrasado'

# Exemplo: qual a probabilidade média de atraso para cada categoria da
#variável dependente, se o indivíduo tiver que percorrer 22km e passar
#por 12 semáforos?

#No nosso exemplo, tempos que:
# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

resultado = modelo_atrasado.predict(pd.DataFrame({'const':[1],
                                                   'dist':[22],
                                                   'sem':[12]})).round(4)

resultado

#Uma maneira de identificar a classe do resultado de acordo com o 'predict'

resultado.idxmax(axis=1)


# In[ ]: Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem, bem como a respectiva
#classificação, ao dataframe original

#Probabilidades de ocorrência das três categoriais
#Definição do array 'phats':
phats = modelo_atrasado.predict()
phats

#Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

#Concatenando o dataframe original com o dataframe 'phats':
df_atrasado_multinomial = pd.concat([df_atrasado_multinomial, phats], axis=1)
df_atrasado_multinomial

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

#Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_atrasado_multinomial['predicao'] = predicao
df_atrasado_multinomial

#Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==0,
                            'predicao_label'] ='não chegou atrasado'
df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==1,
                            'predicao_label'] ='chegou atrasado primeira aula'
df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==2,
                            'predicao_label'] ='chegou atrasado segunda aula'

df_atrasado_multinomial


# In[ ]: Eficiência global do modelo

#Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_atrasado_multinomial,
                       index=['predicao_label'],
                       columns=['atrasado'],
                       aggfunc='size')

#Substituindo 'NaN' por zero
table = table.fillna(0)
table

#Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

#Eficiência global do modelo
acuracia = table.diagonal().sum()/table.sum()
acuracia


# In[ ]: Plotagens das probabilidades

#Plotagem das smooth probability lines para a variável 'dist'

# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

plt.figure(figsize=(10,10))
sns.regplot(x = df_atrasado_multinomial['dist'],
            y = df_atrasado_multinomial[0],
            ci=False, label='não chegou atrasado', scatter=False,
            order=4, color='darkviolet')
plt.scatter(df_atrasado_multinomial['dist'],
            df_atrasado_multinomial[0], alpha=0.5,
            s=60, color='darkviolet')
sns.regplot(x = df_atrasado_multinomial['dist'],
            y = df_atrasado_multinomial[1],
            ci=False, label='chegou atrasado na primeira aula', scatter=False,
            order=4, color='darkorange')
plt.scatter(df_atrasado_multinomial['dist'],
            df_atrasado_multinomial[1], alpha=0.5,
            s=60, color='darkorange')
sns.regplot(x = df_atrasado_multinomial['dist'],
            y = df_atrasado_multinomial[2],
            ci=False, label='chegou atrasado na segunda aula', scatter=False,
            order=4, color='darkgreen')
plt.scatter(df_atrasado_multinomial['dist'],
            df_atrasado_multinomial[2], alpha=0.5,
            s=60, color='darkgreen')
plt.ylabel('Probabilidades', fontsize=15)
plt.xlabel('Distância Percorrida', fontsize=15)
plt.legend(loc='center left', fontsize=12)
plt.show()


# In[ ]: Plotagens das probabilidades

#Plotagem das smooth probability lines para a variável 'sem'

# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

plt.figure(figsize=(10,10))
sns.regplot(x = df_atrasado_multinomial['sem'],
            y = df_atrasado_multinomial[0],
            ci=False, label='não chegou atrasado', scatter=False,
            order=4, color='darkviolet')
plt.scatter(df_atrasado_multinomial['sem'],
            df_atrasado_multinomial[0], alpha=0.5,
            s=60, color='darkviolet')
sns.regplot(x = df_atrasado_multinomial['sem'],
            y = df_atrasado_multinomial[1],
            ci=False, label='chegou atrasado na primeira aula', scatter=False,
            order=4, color='darkorange')
plt.scatter(df_atrasado_multinomial['sem'],
            df_atrasado_multinomial[1], alpha=0.5,
            s=60, color='darkorange')
sns.regplot(x = df_atrasado_multinomial['sem'],
            y = df_atrasado_multinomial[2],
            ci=False, label='chegou atrasado na segunda aula', scatter=False,
            order=4, color='darkgreen')
plt.scatter(df_atrasado_multinomial['sem'],
            df_atrasado_multinomial[2], alpha=0.5,
            s=60, color='darkgreen')
plt.ylabel('Probabilidades', fontsize=15)
plt.xlabel('Quantidade de Semáforos', fontsize=15)
plt.legend(loc='upper center', fontsize=12)
plt.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Probabilidades de não chegar atrasado (função 'go' do pacote 'plotly')

import plotly.io as pio
pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[0],
    opacity=1, intensity=df_atrasado_multinomial[0], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='não chegou atrasado'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Probabilidades de chegar atrasado à primeira aula (função 'go' do pacote
#'plotly')

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[1],
    opacity=1, intensity=df_atrasado_multinomial[1], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='chegou atrasado à primeira aula'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Probabilidades de chegar atrasado à segunda aula (função 'go' do pacote
#'plotly')

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[2],
    opacity=1, intensity=df_atrasado_multinomial[2], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='chegou atrasado à segunda aula'))

plot_figure.show()


# In[ ]: Visualização das sigmóides tridimensionais em um único gráfico

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[0],
    opacity=1)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

trace_1 = go.Mesh3d(
            x=df_atrasado_multinomial['dist'], 
            y=df_atrasado_multinomial['sem'],
            z=df_atrasado_multinomial[1],
            opacity=1)

plot_figure.add_trace(trace_1)

trace_2 = go.Mesh3d(
            x=df_atrasado_multinomial['dist'], 
            y=df_atrasado_multinomial['sem'],
            z=df_atrasado_multinomial[2],
            opacity=1)


plot_figure.add_trace(trace_2)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='probabilidades'))

plot_figure.show()