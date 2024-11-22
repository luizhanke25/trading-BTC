import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import plotly.express as px

# Configurar a interface do Streamlit
st.title("Bitcoin Trading Strategy")
st.write("Uma aplicação interativa para análise de estratégias de trading com aprendizado de máquina.")

# Carregar dataset padrão
@st.cache_data
def load_default_dataset():
    dataset = pd.read_csv('Bitstamp_AAVEBTC_d.csv', skiprows=1)
    dataset.columns = ["unix", "date", "symbol", "open", "high", "low", "close", "Volume_AAVE", "Volume_BTC"]
    dataset.ffill(inplace=True)
    dataset.drop(columns=['unix', 'date', 'symbol'], inplace=True)
    return dataset

dataset = load_default_dataset()

# Feature Engineering
def moving_average(series, window):
    return series.rolling(window=window).mean()

dataset['short_mavg'] = moving_average(dataset['close'], 10)
dataset['long_mavg'] = moving_average(dataset['close'], 60)

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

dataset['RSI'] = RSI(dataset['close'], 14)
dataset.dropna(inplace=True)

# Criar Sinal de Compra/Venda
dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1, 0)

# Divisão dos Dados
X = dataset[['close', 'RSI', 'short_mavg', 'long_mavg']]
y = dataset['signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Interface para ajuste de hiperparâmetros
st.sidebar.header("Ajuste de Hiperparâmetros")
model_choice = st.sidebar.selectbox("Escolha o modelo", ["Random Forest", "Decision Tree", "Logistic Regression", "KNN"])

if model_choice == "Random Forest":
    st.sidebar.markdown("### Modelo de Random Forest")
    st.sidebar.write("Modelo baseado em várias árvores de decisão para melhorar a acurácia e reduzir overfitting.")
    n_estimators = st.sidebar.slider("Número de árvores", min_value=10, max_value=200, step=10, value=100)
    max_depth = st.sidebar.slider("Profundidade máxima", min_value=1, max_value=50, step=1, value=10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_choice == "Decision Tree":
    st.sidebar.markdown("### Modelo de Decision Tree")
    st.sidebar.write("Modelo que classifica os dados com base em regras de decisão em forma de árvore.")
    max_depth = st.sidebar.slider("Profundidade máxima", min_value=1, max_value=50, step=1, value=10)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif model_choice == "Logistic Regression":
    st.sidebar.markdown("### Modelo de Logistic Regression")
    st.sidebar.write("Modelo linear para prever a probabilidade de classes binárias.")
    model = LogisticRegression()
elif model_choice == "KNN":
    st.sidebar.markdown("### Modelo de KNN")
    st.sidebar.write("Modelo que classifica com base nos dados mais próximos do ponto analisado.")
    n_neighbors = st.sidebar.slider("Número de vizinhos", min_value=1, max_value=20, step=1, value=5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Treinar o modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Resultados
st.subheader("Resultados")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Criar tabela de métricas
metrics_data = {
    "Métrica": ["Acurácia", "F1-Score", "MAE", "MSE", "RMSE", "R²", "MAPE"],
    "Valor": [accuracy, f1, mae, mse, rmse, r2, mape]
}
metrics_df = pd.DataFrame(metrics_data)

# Exibir tabela
st.write("### Tabela de Métricas")
st.dataframe(metrics_df)

# Explicação das métricas
st.write("""
### Interpretação das Métricas
- **Acurácia**: Proporção de previsões corretas em relação ao total.
- **F1-Score**: Métrica balanceada que combina precisão e recall.
- **MAE (Erro Médio Absoluto)**: Diferença média absoluta entre os valores reais e previstos.
- **MSE (Erro Quadrático Médio)**: Penaliza erros grandes devido ao quadrado das diferenças.
- **RMSE (Raiz do Erro Quadrático Médio)**: Representa o erro médio em unidades originais.
- **R²**: Mostra o quanto o modelo explica a variação dos dados (1 é o melhor).
- **MAPE**: Percentual médio do erro em relação aos valores reais (quanto menor, melhor).
""")

# Gráfico comparativo
st.write("### Gráfico Comparativo de Métricas")
fig = px.bar(metrics_df, x="Métrica", y="Valor", title="Comparação de Métricas", text="Valor")
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig)

# Explicação do gráfico
st.write("**Este gráfico compara as métricas calculadas, permitindo identificar o desempenho geral do modelo selecionado.**")

# Matriz de Confusão
st.subheader("Matriz de Confusão")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

st.write("**A matriz de confusão apresenta as previsões corretas e incorretas para cada classe, permitindo avaliar o desempenho do modelo em detalhes.**")

# Visualização da Árvore de Decisão (se aplicável)
if model_choice == "Decision Tree":
    st.subheader("Visualização da Árvore de Decisão")
    tree_depth = st.sidebar.slider("Profundidade da Árvore a ser exibida", min_value=1, max_value=max_depth, step=1, value=3)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, max_depth=tree_depth, feature_names=X.columns, class_names=["Venda", "Compra"], filled=True, ax=ax)
    st.pyplot(fig)
    st.write("**Esta é a representação gráfica da árvore de decisão treinada, mostrando como o modelo toma decisões em cada etapa.**")

# Backtesting
st.subheader("Backtesting")
dataset['Market Returns'] = dataset['close'].pct_change()
dataset['Strategy Returns'] = dataset['Market Returns'] * dataset['signal'].shift(1)
cumulative_strategy_returns = (1 + dataset['Strategy Returns']).cumprod()
cumulative_market_returns = (1 + dataset['Market Returns']).cumprod()

fig = px.line(dataset, y=["Market Returns", "Strategy Returns"], title="Retornos Acumulados")
st.plotly_chart(fig)

st.write("**O gráfico acima compara os retornos acumulados da estratégia implementada com os retornos do mercado.**")

# Tabela Interativa
st.subheader("Visualizar Dados")
st.dataframe(dataset.head(20))
