# 🎯 DESAFIO II - Ciência e Governança de Dados
## Modelagem Preditiva do IDHM e Recomendações Estratégicas (Gradient Boosting)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Índice

- [Objetivo do Projeto](#objetivo-do-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Dataset e Variáveis](#dataset-e-variáveis)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Recomendações Estratégicas](#recomendações-estratégicas)
- [Como Executar](#como-executar)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Autor](#autor)

---

## 🎯 Objetivo do Projeto

**Pergunta Central**: *Como poderíamos avaliar e prever/visualizar os agentes/fenômenos que mais causam impactos socioeconômicos no Brasil?*

Este projeto tem como objetivo desenvolver modelos preditivos para avaliar o **Índice de Desenvolvimento Humano Municipal (IDHM)** e identificar os principais fatores socioeconômicos que influenciam o desenvolvimento humano no Brasil.

### Objetivos Específicos:

1. ✅ Desenvolver modelos de Machine Learning para predição do IDHM
2. ✅ Comparar múltiplos modelos e avaliar desempenho
3. ✅ Identificar as variáveis mais importantes através de análise SHAP
4. ✅ Criar visualizações interativas dos resultados
5. ✅ Formular recomendações estratégicas baseadas nos insights obtidos

---

## 📁 Estrutura do Repositório

```
zetta-lab2/
│
├── data/
│   └── refined/
│       └── base_udh_refined.csv          # Dataset processado (1228 obs, 13 vars)
│
├── notebooks/
│   ├── eda_outliers_nulos.ipynb          # Análise Exploratória de Dados
│   ├── etl_refined.ipynb                 # ETL e Preparação dos Dados
│   └── model.ipynb                       # Modelagem Gradient Boosting + SHAP
│
├── scripts/
│   ├── model_comparison.py               # Comparação de múltiplos modelos
│   └── toCSV.py                          # Utilitário de conversão
│
├── outputs/
│   ├── boxplot_outliers.jpg              # Análise de outliers
│   ├── shap_summary.jpg                  # SHAP Summary Plot (GB)
│   ├── shap_local_explanation.jpg        # Explicação SHAP local (GB)
│   ├── shap_importance_results.csv       # Importância das features (GB)
│   ├── model_comparison_results.csv      # Resultados comparativos
│   ├── model_comparison_metrics.jpg      # Visualizações comparativas
│   └── model_r2_train_vs_test.jpg        # Análise de generalização
│
├── dashboard/
│   └── app.py                            # Dashboard interativo (Streamlit/Plotly)
│
├── README.md                             # Este arquivo
└── requirements.txt                      # Dependências do projeto
```

---

## 📊 Dataset e Variáveis

### Fonte dos Dados

Os dados utilizados foram obtidos do **Atlas do Desenvolvimento Humano no Brasil** e correspondem ao IDHM de municípios brasileiros.

### Dimensões do Dataset

- **Observações**: 1.228 municípios
- **Variáveis**: 13 (12 features + 1 target)
- **Período**: Dados do último censo disponível
- **Valores ausentes**: 18 linhas removidas durante o ETL

> **Observação**: O dataset refinado contém apenas variáveis numéricas (não inclui município/UF/IBGE), portanto o foco é em análise global e interpretabilidade.

### Variáveis do Modelo

#### Variável Target (Dependente)

| Variável | Descrição | Tipo |
|----------|-----------|------|
| **IDHM** | Índice de Desenvolvimento Humano Municipal | Float (0-1) |

#### Features (Variáveis Independentes)

| Variável | Descrição | Média | Desvio Padrão |
|----------|-----------|-------|---------------|
| **T_ANALF15M** | Taxa de analfabetismo (15 anos ou mais) | 6.55% | 4.15% |
| **T_ATRASO_2_BASICO** | Taxa de atraso escolar 2+ anos | 18.35% | 5.97% |
| **T_FUND18M** | Taxa sem fundamental completo (18+ anos) | 49.73% | 18.39% |
| **AGUA_ESGOTO** | População sem água encanada e esgoto | 1.50% | 2.79% |
| **T_DENS** | Taxa de densidade demográfica | 30.12 | 13.88 |
| **T_LIXO** | Coleta de lixo adequada | 95.09% | 9.64% |
| **GINI** | Índice de Gini (desigualdade) | 0.44 | 0.05 |
| **PPOB** | Percentual de pobres | 32.21% | 18.85% |
| **T_FUNDIN18MINF** | Taxa sem fundamental (18 anos inf.) | 34.35% | 13.21% |
| **P_FORMAL** | Grau de formalização | 67.98% | 6.67% |
| **T_DES18M** | Taxa de desemprego (18+ anos) | 12.30% | 6.33% |
| **RAZDEP** | Razão de dependência | 46.13 | 8.96 |

---

## 🔬 Metodologia

### 1. Preparação dos Dados 

#### ETL Pipeline
- **Importação**: Leitura do dataset bruto
- **Limpeza**: Remoção de 18 linhas com valores ausentes
- **Tratamento de Outliers**: Identificação via IQR (358 outliers detectados)
- **Decisão**: Manutenção dos outliers por representarem municípios reais
- **Normalização**: Padronização aplicada quando necessário

#### Análise Exploratória (EDA)
- Análise de distribuições (boxplots)
- Identificação de correlações
- Detecção de padrões e anomalias
- Visualização de relações entre variáveis

### 2. Modelagem de Machine Learning

#### Divisão dos Dados
```python
- Treino: 80% (982 observações)
- Teste: 20% (246 observações)
- Random State: 42 (reprodutibilidade)
```

#### Modelos Comparados

| Modelo | Tipo | Justificativa |
|--------|------|---------------|
| **Linear Regression** | Baseline | Referência para modelos complexos |
| **Ridge** | Regularização L2 | Controle de overfitting |
| **Lasso** | Regularização L1 | Seleção de features |
| **ElasticNet** | Regularização L1+L2 | Combinação de Ridge e Lasso |
| **Decision Tree** | Árvore de decisão | Interpretabilidade |
| **Random Forest** | Ensemble (Bagging) | Comparativo robusto |
| **Gradient Boosting** | Ensemble (Boosting) | **MODELO ESCOLHIDO** ⭐ |
| **AdaBoost** | Ensemble (Boosting) | Redução de viés |
| **KNN** | Instância | Aprendizado por vizinhança |
| **SVR** | Kernel | Relações não-lineares |

#### Otimização de Hiperparâmetros

**Gradient Boosting - Grid Search (5-Fold CV)**

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [2, 3, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

**Melhores Hiperparâmetros Encontrados:**
- Determinados via GridSearchCV no notebook `model.ipynb`
- Otimização baseada em R² Score com validação cruzada

### 3. Análise de Importância das Variáveis (SHAP)

Utilizamos **SHAP (SHapley Additive exPlanations)** para interpretabilidade do modelo Gradient Boosting:

- **SHAP Values**: Valores de Shapley para cada predição
- **Summary Plot**: Impacto global das features
- **Waterfall Plot**: Explicação local para instâncias específicas
- **Feature Importance**: Ranking de importância baseado em |SHAP|

---

## 📈 Resultados

### Desempenho do Modelo Gradient Boosting (Campeão)

#### Métricas de Avaliação

| Métrica | Treino | Teste |
|---------|--------|-------|
| **R² Score** | 0.9999 | **0.9973** ⭐ |
| **MAE** | - | **0.0028** |
| **RMSE** | - | **0.0057** |
| **Overfitting** | - | **0.0026** |

#### Interpretação das Métricas

✅ **Excelente Generalização**: Diferença mínima entre R² treino e teste (0.0026)  
✅ **Alta Precisão**: MAE de 0.0028 significa erro médio de apenas 0.28 pontos percentuais  
✅ **Baixo Overfitting**: Modelo não está sobreajustado aos dados de treino  
✅ **Melhor Desempenho**: Gradient Boosting superou todos os outros modelos testados  

### Importância das Variáveis (SHAP)

#### Top 5 Features Mais Importantes

| Ranking | Variável | SHAP Value Médio | Interpretação |
|---------|----------|------------------|---------------|
| 🥇 **1º** | **T_FUND18M** | 0.026032 | Taxa sem fundamental completo (18+ anos) |
| 🥈 **2º** | **PPOB** | 0.018661 | Percentual de pobres no município |
| 🥉 **3º** | **T_FUNDIN18MINF** | 0.018213 | Taxa sem fundamental (18 anos inf.) |
| **4º** | **T_DENS** | 0.011604 | Densidade demográfica |
| **5º** | **T_ATRASO_2_BASICO** | 0.006096 | Taxa de atraso escolar 2+ anos |

#### Insights do SHAP Summary Plot

1. **Educação é fundamental**: As 3 variáveis educacionais (T_FUND18M, T_FUNDIN18MINF, T_ATRASO_2_BASICO) dominam a importância
2. **Pobreza tem impacto direto**: PPOB (2ª posição) mostra correlação forte com IDHM
3. **Urbanização importa**: T_DENS indica que densidade populacional influencia desenvolvimento
4. **Direção dos impactos**:
   - 🔴 **Valores altos** de educação precária → **Reduzem IDHM**
   - 🔵 **Valores baixos** de educação precária → **Aumentam IDHM**

### Comparação de Modelos (Top 5)

| Modelo | R² Test | MAE | RMSE | Overfitting |
|--------|---------|-----|------|-------------|
| **Gradient Boosting** ⭐ | **0.9973** | **0.0028** | **0.0057** | **0.0026** |
| Random Forest | 0.9966 | 0.0033 | 0.0063 | 0.0029 |
| Decision Tree | 0.9934 | 0.0034 | 0.0088 | 0.0065 |
| Linear Regression | 0.9928 | 0.0071 | 0.0092 | -0.0014 |
| Ridge | 0.9917 | 0.0077 | 0.0099 | -0.0009 |

**Conclusão**: Gradient Boosting apresentou o melhor desempenho geral, com R² Test superior e excelente controle de overfitting.

---

## 💡 Recomendações Estratégicas

Baseado nos insights do modelo e análise SHAP, recomendamos:

### 🎓 1. EDUCAÇÃO (Prioridade Máxima)

**Problema Identificado**: Variáveis educacionais são os principais determinantes do IDHM (ocupam 3 das 5 primeiras posições no ranking SHAP)

**Recomendações**:

✅ **Meta 1**: Reduzir taxa de pessoas sem fundamental completo (T_FUND18M)
- Implementar programas de EJA (Educação de Jovens e Adultos)
- Criar incentivos financeiros para conclusão do ensino fundamental
- Estabelecer parcerias com empresas para educação corporativa

✅ **Meta 2**: Combater atraso escolar (T_ATRASO_2_BASICO)
- Programa de reforço escolar em municípios críticos
- Acompanhamento individualizado de alunos em risco
- Capacitação de professores para ensino personalizado

✅ **Meta 3**: Ampliar acesso à educação infantil
- Construção de creches e pré-escolas em áreas prioritárias
- Subsídios para famílias de baixa renda
- Programas de desenvolvimento na primeira infância

**Impacto Esperado**: Aumento de 0.05-0.08 pontos no IDHM em 10 anos

---

### 💰 2. COMBATE À POBREZA

**Problema Identificado**: PPOB (2ª variável mais importante) com forte impacto negativo no IDHM

**Recomendações**:

✅ **Transferência de renda**
- Ampliar cobertura de programas sociais
- Revisão de critérios para incluir vulneráveis não cadastrados
- Integração de bancos de dados governamentais

✅ **Geração de emprego e renda**
- Incentivos fiscais para empresas em regiões pobres
- Programas de microcrédito e empreendedorismo
- Cursos profissionalizantes alinhados ao mercado local

✅ **Desenvolvimento local**
- Fortalecimento de cooperativas e associações
- Apoio à agricultura familiar
- Turismo comunitário em regiões com potencial

**Impacto Esperado**: Redução de 5-10% na taxa de pobreza em 5 anos

---

### 🏙️ 3. INFRAESTRUTURA URBANA E DESENVOLVIMENTO TERRITORIAL

**Problema Identificado**: Densidade demográfica (T_DENS) influencia significativamente o IDHM (4ª posição)

**Recomendações**:

✅ **Municípios de baixa densidade**
- Investir em conectividade (internet, estradas)
- Telemedicina e ensino à distância
- Incentivos para fixação de profissionais qualificados

✅ **Municípios de alta densidade**
- Planejamento urbano para áreas metropolitanas
- Transporte público eficiente
- Habitação social e regularização fundiária

✅ **Saneamento básico**
- Priorizar municípios com alta taxa de AGUA_ESGOTO
- Parcerias público-privadas para universalização
- Educação ambiental e sanitária

**Impacto Esperado**: Melhoria de 0.02-0.04 pontos no IDHM em 8 anos

---

### 📊 4. POLÍTICAS BASEADAS EM DADOS

**Recomendação Transversal**:

✅ **Sistema de Monitoramento**
- Dashboard em tempo real com indicadores do IDHM
- Alertas automáticos para municípios em declínio
- Benchmark entre municípios similares

✅ **Alocação Inteligente de Recursos**
- Utilizar modelo preditivo para priorizar investimentos
- Simular impacto de políticas antes da implementação
- Avaliação contínua de efetividade das ações

✅ **Transparência e Accountability**
- Publicação trimestral de resultados
- Ranking de municípios por evolução no IDHM
- Premiação de boas práticas

---

### 🎯 5. METAS QUANTITATIVAS (2026-2036)

| Indicador | Situação Atual | Meta 2036 | Δ Esperado |
|-----------|----------------|-----------|------------|
| **IDHM Médio Nacional** | 0.684 | **0.750** | +0.066 |
| **Taxa sem Fundamental (T_FUND18M)** | 49.7% | **< 35%** | -14.7 pp |
| **Percentual de Pobres (PPOB)** | 32.2% | **< 20%** | -12.2 pp |
| **Municípios IDHM > 0.7** | ~50% | **> 80%** | +30 pp |

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.12+
- pip ou conda

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/AlencarAvelar/zetta-lab2
cd zetta-lab2

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Executar Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir notebooks na ordem:
# 1. eda_outliers_nulos.ipynb
# 2. etl_refined.ipynb
# 3. model.ipynb
```

### Executar Script de Comparação

```bash
cd scripts
python model_comparison.py
```

### Executar Dashboard

```bash
cd dashboard
streamlit run app.py
# Abrir navegador em http://localhost:8501
```

---

## 🛠️ Tecnologias Utilizadas

### Linguagem e Ambiente
- **Python 3.12+**
- **Jupyter Notebook**

### Bibliotecas de Data Science
- **pandas** 2.0+ - Manipulação de dados
- **numpy** 1.24+ - Computação numérica
- **scikit-learn** 1.3+ - Machine Learning

### Visualização
- **matplotlib** 3.7+ - Gráficos estáticos
- **seaborn** 0.12+ - Visualizações estatísticas
- **plotly** 5.14+ - Gráficos interativos
- **streamlit** 1.24+ - Dashboard web

### Interpretabilidade
- **shap** 0.41+ - Explicabilidade de modelos

### Controle de Versão
- **Git & GitHub** - Versionamento e colaboração

---

## 👤 Autor

**Alencara Avelar**  
📧 Email: alencarhlavelar@gmail.com  
🔗 LinkedIn: [linkedin.com/in/alencar-avelar-a712591b7](https://www.linkedin.com/in/alencar-avelar-a712591b7/)  
🐙 GitHub: [github.com/AlencarAvelar](https://github.com/AlencarAvelar)

---
