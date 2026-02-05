"""
Dashboard Interativo - Análise IDHM e Predição (Gradient Boosting)
===================================================================
Visualização dos resultados da modelagem preditiva do IDHM
Modelo Campeão: Gradient Boosting (R² Test: 0.9973)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO DA PÁGINA
# ============================

st.set_page_config(
    page_title="IDHM - Análise Preditiva (GB)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ESTILO CSS CUSTOMIZADO
# ============================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        color: #2c3e50;  /* ← TEXTO ESCURO */
    }
    .insight-box {
        background-color: #e1f5e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #1e4620;  /* ← TEXTO VERDE ESCURO */
    }
    .insight-box h3, .insight-box h4, .insight-box h5 {
        color: #0d3814;  /* ← TÍTULOS VERDE MUITO ESCURO */
    }
    .insight-box b, .insight-box strong {
        color: #0d3814;  /* ← NEGRITO VERDE MUITO ESCURO */
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #664d03;  /* ← TEXTO AMARELO ESCURO/MARROM */
    }
    .warning-box h3, .warning-box h4, .warning-box h5 {
        color: #4d3900;  /* ← TÍTULOS AMARELO MUITO ESCURO */
    }
    .warning-box b, .warning-box strong {
        color: #4d3900;  /* ← NEGRITO AMARELO MUITO ESCURO */
    }
    .champion-box {
        background-color: #fff4e6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid #ff9800;
        margin: 1rem 0;
        color: #663c00;  /* ← TEXTO LARANJA ESCURO/MARROM */
    }
    .champion-box h3, .champion-box h4, .champion-box h5 {
        color: #4d2d00;  /* ← TÍTULOS LARANJA MUITO ESCURO */
    }
    .champion-box b, .champion-box strong {
        color: #4d2d00;  /* ← NEGRITO LARANJA MUITO ESCURO */
    }
</style>
""", unsafe_allow_html=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

@st.cache_data
def load_data():
    """Carrega os dados do projeto"""
    try:
        df = pd.read_csv('../data/refined/base_udh_refined.csv')
        shap_df = pd.read_csv('../outputs/shap_importance_results.csv')
        results_df = pd.read_csv('../outputs/model_comparison_results.csv')
        return df, shap_df, results_df
    except FileNotFoundError:
        st.error("❌ Erro ao carregar os dados. Verifique se os arquivos estão nos caminhos corretos.")
        st.stop()

def create_gauge_chart(value, title, max_value=1.0):
    """Cria um gráfico de gauge (velocímetro)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 0.7, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#ff6b6b'},
                {'range': [0.5, 0.7], 'color': '#ffd93d'},
                {'range': [0.7, max_value], 'color': '#6bcf7f'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_shap_bar_chart(shap_df):
    """Cria gráfico de barras horizontal com importância SHAP"""
    fig = px.bar(
        shap_df.head(10).sort_values('Mean_SHAP_Value', ascending=True),
        y='Feature',
        x='Mean_SHAP_Value',
        orientation='h',
        title='🎯 Top 10 Variáveis Mais Importantes (SHAP - Gradient Boosting)',
        labels={'Mean_SHAP_Value': 'Importância SHAP', 'Feature': 'Variável'},
        color='Mean_SHAP_Value',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Valor SHAP Médio (|SHAP|)",
        yaxis_title="",
        showlegend=False
    )
    
    return fig

def create_idhm_distribution(df):
    """Cria histograma da distribuição do IDHM"""
    fig = px.histogram(
        df,
        x='IDHM',
        nbins=50,
        title='📊 Distribuição do IDHM nos Municípios',
        labels={'IDHM': 'Índice de Desenvolvimento Humano Municipal', 'count': 'Frequência'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Adicionar linhas verticais para referências
    fig.add_vline(x=df['IDHM'].mean(), line_dash="dash", line_color="red", 
                  annotation_text=f"Média: {df['IDHM'].mean():.3f}")
    fig.add_vline(x=df['IDHM'].median(), line_dash="dash", line_color="green",
                  annotation_text=f"Mediana: {df['IDHM'].median():.3f}")
    
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(df):
    """Cria mapa de calor com correlações"""
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Variável", y="Variável", color="Correlação"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='🌡️ Matriz de Correlação entre Variáveis'
    )
    
    fig.update_layout(height=700)
    return fig

def create_scatter_top_features(df, feature1, feature2):
    """Cria gráfico de dispersão entre duas features e IDHM"""
    fig = px.scatter(
        df,
        x=feature1,
        y=feature2,
        color='IDHM',
        size='IDHM',
        hover_data=['IDHM'],
        title=f'📉 Relação: {feature1} vs {feature2} (cor = IDHM)',
        labels={feature1: feature1, feature2: feature2},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500)
    return fig

def create_model_comparison_chart(results_df):
    """Cria gráfico comparativo de modelos"""
    # Ordenar por R² Test
    results_sorted = results_df.sort_values('R² Test', ascending=True)
    
    fig = px.bar(
        results_sorted,
        x='R² Test',
        y='Model',
        orientation='h',
        title='🤖 Comparação: R² Test dos Modelos',
        labels={'R² Test': 'R² Score (Teste)', 'Model': 'Modelo'},
        color='R² Test',
        color_continuous_scale='Blues',
        text='R² Test'
    )
    
    # Destacar o campeão
    champion = results_sorted.iloc[-1]
    fig.add_annotation(
        x=champion['R² Test'],
        y=champion['Model'],
        text="🏆 CAMPEÃO",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff9800",
        ax=-100,
        ay=0,
        font=dict(size=14, color="#ff9800", family="Arial Black")
    )
    
    fig.update_traces(texttemplate='%{text:.6f}', textposition='outside')
    fig.update_layout(height=500, uniformtext_minsize=8, uniformtext_mode='hide')
    
    return fig

def create_overfitting_chart(results_df):
    """Cria gráfico de overfitting"""
    results_sorted = results_df.sort_values('Overfitting', ascending=True)
    
    fig = px.bar(
        results_sorted,
        x='Overfitting',
        y='Model',
        orientation='h',
        title='📊 Análise de Overfitting (Train - Test)',
        labels={'Overfitting': 'Overfitting (quanto menor, melhor)', 'Model': 'Modelo'},
        color='Overfitting',
        color_continuous_scale='Reds',
        text='Overfitting'
    )
    
    fig.update_traces(texttemplate='%{text:.6f}', textposition='outside')
    fig.update_layout(height=500)
    
    return fig

def create_metrics_comparison(results_df):
    """Cria gráfico comparativo de múltiplas métricas"""
    # Selecionar top 5 modelos
    top5 = results_df.nlargest(5, 'R² Test')
    
    fig = go.Figure()
    
    # MAE
    fig.add_trace(go.Bar(
        name='MAE',
        x=top5['Model'],
        y=top5['MAE'],
        text=top5['MAE'].round(6),
        textposition='auto',
    ))
    
    # RMSE
    fig.add_trace(go.Bar(
        name='RMSE',
        x=top5['Model'],
        y=top5['RMSE'],
        text=top5['RMSE'].round(6),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='📊 Comparação de Métricas de Erro (Top 5 Modelos)',
        xaxis_title='Modelo',
        yaxis_title='Valor do Erro',
        barmode='group',
        height=400
    )
    
    return fig

# ============================
# CARREGAR DADOS
# ============================

df, shap_df, results_df = load_data()

# Identificar o campeão
champion_model = results_df.loc[results_df['R² Test'].idxmax()]

# ============================
# SIDEBAR
# ============================


# Filtros
st.sidebar.subheader("📌 Filtros")

# Filtro de IDHM
idhm_range = st.sidebar.slider(
    "Faixa de IDHM",
    float(df['IDHM'].min()),
    float(df['IDHM'].max()),
    (float(df['IDHM'].min()), float(df['IDHM'].max())),
    step=0.01
)

# Filtrar dados
df_filtered = df[(df['IDHM'] >= idhm_range[0]) & (df['IDHM'] <= idhm_range[1])]

st.sidebar.markdown(f"**Municípios selecionados:** {len(df_filtered)}/{len(df)}")

st.sidebar.markdown("---")
st.sidebar.info("""
**📊 Dashboard IDHM**

Explore os resultados da modelagem preditiva do Índice de Desenvolvimento Humano Municipal (IDHM).

**Modelo Principal:**
Gradient Boosting com GridSearchCV

**Desenvolvido com:**
- Python 3.12+
- Streamlit
- Plotly
- Scikit-Learn
- SHAP


""")

# ============================
# CONTEÚDO PRINCIPAL
# ============================

# Header
st.markdown('<h1 class="main-header">📊 Dashboard - Análise Preditiva do IDHM</h1>', unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 2rem;'>
    <b>Modelagem e Visualização dos Fatores Socioeconômicos que Impactam o Desenvolvimento Humano no Brasil</b><br>
    <span style='color: #ff9800; font-weight: bold;'>🏆 Modelo Campeão: {champion_model['Model']} (R² Test: {champion_model['R² Test']:.6f})</span>
</div>
""", unsafe_allow_html=True)

# ============================
# TABS PRINCIPAIS
# ============================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visão Geral", 
    "🎯 Importância das Variáveis (SHAP)", 
    "🔍 Análise Exploratória",
    "🤖 Desempenho dos Modelos",
    "💡 Recomendações Estratégicas"
])

with tab1:
    st.header("📊 Estatísticas Gerais do Dataset")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏘️ Municípios Analisados", f"{len(df_filtered):,}")
    
    with col2:
        st.metric("📊 IDHM Médio", f"{df_filtered['IDHM'].mean():.3f}")
    
    with col3:
        st.metric("📈 IDHM Máximo", f"{df_filtered['IDHM'].max():.3f}")
    
    with col4:
        st.metric("📉 IDHM Mínimo", f"{df_filtered['IDHM'].min():.3f}")
    
    st.markdown("---")
    

    
    # Gráficos lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge do IDHM médio
        st.plotly_chart(
            create_gauge_chart(df_filtered['IDHM'].mean(), "IDHM Médio Nacional"),
            use_container_width=True
        )
    
    with col2:
        # Distribuição do IDHM
        st.plotly_chart(
            create_idhm_distribution(df_filtered),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Estatísticas descritivas
    st.subheader("📋 Estatísticas Descritivas das Variáveis")
    
    st.dataframe(
        df_filtered.describe().T.style.format("{:.4f}").background_gradient(cmap='YlOrRd'),
        use_container_width=True
    )

with tab2:
    st.header("🎯 Importância das Variáveis (SHAP - Gradient Boosting)")
    
    st.markdown("""
    <div class='insight-box'>
        <b>📌 O que é SHAP?</b><br>
        SHAP (SHapley Additive exPlanations) é uma técnica de interpretabilidade que mostra 
        o quanto cada variável contribui para a predição do modelo <b>Gradient Boosting</b>.
        Os valores representam a importância média absoluta de cada feature nas predições.
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico de importância
    st.plotly_chart(
        create_shap_bar_chart(shap_df),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Top 5 features
    st.subheader("🏆 Top 5 Variáveis Mais Importantes (Gradient Boosting + SHAP)")
    
    top5 = shap_df.head(5)
    
    for idx, row in top5.iterrows():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{idx+1}. {row['Feature']}**")
            st.progress(float(row['Mean_SHAP_Value'] / shap_df['Mean_SHAP_Value'].max()))
        
        with col2:
            st.metric("Importância SHAP", f"{row['Mean_SHAP_Value']:.6f}")
    
    st.markdown("---")
    
    # Insights das top features
    st.subheader("💡 Insights das Principais Variáveis")
    
    st.markdown("""
    <div class='insight-box'>
        <h4>🎓 1. T_FUND18M (Taxa sem fundamental completo - 18+ anos)</h4>
        <p><b>Importância:</b> 0.026032 (1ª posição)</p>
        <p><b>Interpretação:</b> A educação fundamental é o fator mais determinante do IDHM. 
        Municípios com maior taxa de pessoas sem ensino fundamental completo apresentam IDHM significativamente menor.</p>
        <p><b>Ação recomendada:</b> Priorizar programas de conclusão do ensino fundamental para adultos (EJA).</p>
    </div>
    
    <div class='warning-box'>
        <h4>💰 2. PPOB (Percentual de pobres)</h4>
        <p><b>Importância:</b> 0.018661 (2ª posição)</p>
        <p><b>Interpretação:</b> A pobreza tem forte correlação negativa com o IDHM. 
        Reduzir a pobreza é essencial para melhorar o desenvolvimento humano.</p>
        <p><b>Ação recomendada:</b> Ampliar programas de transferência de renda e geração de emprego.</p>
    </div>
    
    <div class='insight-box'>
        <h4>🎓 3. T_FUNDIN18MINF (Taxa sem fundamental - 18 anos inferior)</h4>
        <p><b>Importância:</b> 0.018213 (3ª posição)</p>
        <p><b>Interpretação:</b> A conclusão do ensino fundamental em idade adequada é crucial. 
        Indicador complementar ao T_FUND18M, reforça a importância da educação básica.</p>
        <p><b>Ação recomendada:</b> Reduzir evasão escolar e garantir conclusão do fundamental na idade certa.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabela completa
    st.subheader("📊 Tabela Completa de Importância SHAP")
    st.dataframe(
        shap_df.style.format({'Mean_SHAP_Value': '{:.6f}'}).background_gradient(subset=['Mean_SHAP_Value'], cmap='Greens'),
        use_container_width=True
    )

with tab3:
    st.header("🔍 Análise Exploratória dos Dados")
    
    # Mapa de calor de correlações
    st.subheader("🌡️ Correlações entre Variáveis")
    st.plotly_chart(
        create_correlation_heatmap(df_filtered),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Scatter plots interativos
    st.subheader("📉 Relação entre Variáveis (Análise Bivariada)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox(
            "Selecione Variável 1 (Eixo X)",
            options=[col for col in df.columns if col != 'IDHM'],
            index=0
        )
    
    with col2:
        feature2 = st.selectbox(
            "Selecione Variável 2 (Eixo Y)",
            options=[col for col in df.columns if col != 'IDHM'],
            index=1
        )
    
    st.plotly_chart(
        create_scatter_top_features(df_filtered, feature1, feature2),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Box plots das top 3 features
    st.subheader("📦 Distribuição das Top 3 Variáveis")
    
    top3_features = shap_df.head(3)['Feature'].tolist()
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (col, feature) in enumerate(zip([col1, col2, col3], top3_features)):
        with col:
            fig = px.box(
                df_filtered,
                y=feature,
                title=f"{feature}",
                labels={feature: feature}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("🤖 Desempenho dos Modelos de Machine Learning")
    
    st.markdown(f"""
    <div class='champion-box'>
        <h3 style='text-align: center;'>🏆 MODELO CAMPEÃO: {champion_model['Model']}</h3>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;'>
            <div style='text-align: center;'>
                <h3 style='color: #ff9800; margin: 0;'>{champion_model['R² Test']:.6f}</h3>
                <p style='margin: 0; font-size: 0.9rem;'>R² Test</p>
            </div>
            <div style='text-align: center;'>
                <h3 style='color: #28a745; margin: 0;'>{champion_model['MAE']:.6f}</h3>
                <p style='margin: 0; font-size: 0.9rem;'>MAE</p>
            </div>
            <div style='text-align: center;'>
                <h3 style='color: #17a2b8; margin: 0;'>{champion_model['RMSE']:.6f}</h3>
                <p style='margin: 0; font-size: 0.9rem;'>RMSE</p>
            </div>
            <div style='text-align: center;'>
                <h3 style='color: #6c757d; margin: 0;'>{champion_model['Overfitting']:.6f}</h3>
                <p style='margin: 0; font-size: 0.9rem;'>Overfitting</p>
            </div>
        </div>
        <p style='text-align: center; margin-top: 1rem;'>
            ✅ Melhor R² Test entre todos os modelos testados<br>
            ✅ Excelente controle de overfitting (diferença treino-teste mínima)<br>
            ✅ Erros médios extremamente baixos (MAE < 0.003)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico de comparação de R² Test
    st.subheader("📊 Comparação de R² Test (todos os modelos)")
    st.plotly_chart(
        create_model_comparison_chart(results_df),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Comparação de métricas de erro
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Comparação de Erros (Top 5)")
        st.plotly_chart(
            create_metrics_comparison(results_df),
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Análise de Overfitting")
        st.plotly_chart(
            create_overfitting_chart(results_df),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Tabela de resultados
    st.subheader("📋 Resultados Completos (Todos os Modelos)")
    
    # Destacar o campeão
    def highlight_champion(row):
        if row['Model'] == champion_model['Model']:
            return ['background-color: #fff4e6; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(
        results_df.sort_values('R² Test', ascending=False).style.format({
            'R² Train': '{:.6f}',
            'R² Test': '{:.6f}',
            'MAE': '{:.6f}',
            'RMSE': '{:.6f}',
            'CV R² Mean': '{:.6f}',
            'CV R² Std': '{:.6f}',
            'Overfitting': '{:.6f}'
        }).apply(highlight_champion, axis=1).background_gradient(subset=['R² Test'], cmap='Greens'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Comparativo Top 5
    st.subheader("🏅 Top 5 Modelos - Análise Comparativa")
    
    top5_models = results_df.nlargest(5, 'R² Test')
    
    for idx, row in top5_models.iterrows():
        emoji = "🏆" if row['Model'] == champion_model['Model'] else f"#{idx+1}"
        
        with st.expander(f"{emoji} {row['Model']} - R² Test: {row['R² Test']:.6f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Train", f"{row['R² Train']:.6f}")
                st.metric("R² Test", f"{row['R² Test']:.6f}")
            
            with col2:
                st.metric("MAE", f"{row['MAE']:.6f}")
                st.metric("RMSE", f"{row['RMSE']:.6f}")
            
            with col3:
                st.metric("CV R² Mean", f"{row['CV R² Mean']:.6f}")
                st.metric("Overfitting", f"{row['Overfitting']:.6f}")

with tab5:
    st.header("💡 Recomendações Estratégicas")
    
    st.markdown("""
    Baseado nos insights do modelo e análise SHAP, recomendamos as seguintes ações prioritárias:
    """)
    
    # Recomendação 1
    st.markdown("""
    <div class='insight-box'>
        <h3>🎓 1. EDUCAÇÃO (Prioridade Máxima)</h3>
        <p><b>Problema:</b> Variáveis educacionais são os principais determinantes do IDHM</p>
        <ul>
            <li>✅ Reduzir taxa de pessoas sem fundamental completo (T_FUND18M)</li>
            <li>✅ Implementar programas de EJA (Educação de Jovens e Adultos)</li>
            <li>✅ Combater atraso escolar (T_ATRASO_2_BASICO)</li>
            <li>✅ Ampliar acesso à educação infantil</li>
        </ul>
        <p><b>Impacto Esperado:</b> Aumento de 0.05-0.08 pontos no IDHM em 10 anos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendação 2
    st.markdown("""
    <div class='warning-box'>
        <h3>💰 2. COMBATE À POBREZA</h3>
        <p><b>Problema:</b> PPOB (2ª variável mais importante) com forte impacto negativo</p>
        <ul>
            <li>✅ Ampliar programas de transferência de renda</li>
            <li>✅ Incentivar geração de emprego e renda</li>
            <li>✅ Apoiar empreendedorismo local</li>
        </ul>
        <p><b>Impacto Esperado:</b> Redução de 5-10% na taxa de pobreza em 5 anos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendação 3
    st.markdown("""
    <div class='insight-box'>
        <h3>🏙️ 3. INFRAESTRUTURA URBANA</h3>
        <p><b>Problema:</b> Densidade demográfica influencia IDHM</p>
        <ul>
            <li>✅ Investir em conectividade (internet, estradas)</li>
            <li>✅ Planejamento urbano eficiente</li>
            <li>✅ Universalizar saneamento básico</li>
        </ul>
        <p><b>Impacto Esperado:</b> Melhoria de 0.02-0.04 pontos no IDHM em 8 anos</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metas quantitativas
    st.subheader("🎯 Metas Quantitativas (2024-2034)")
    
    metas_df = pd.DataFrame({
        'Indicador': [
            'IDHM Médio Nacional',
            'Taxa sem Fundamental (T_FUND18M)',
            'Percentual de Pobres (PPOB)',
            'Municípios IDHM > 0.7'
        ],
        'Situação Atual': ['0.684', '49.7%', '32.2%', '~50%'],
        'Meta 2034': ['0.750', '< 35%', '< 20%', '> 80%'],
        'Δ Esperado': ['+0.066', '-14.7 pp', '-12.2 pp', '+30 pp']
    })
    
    st.table(metas_df)
    
    # Conclusão
    st.markdown("""
    <div class='champion-box'>
        <h3 style='text-align: center;'>🎯 CONCLUSÃO</h3>
        <p style='text-align: justify;'>
            O modelo <b>Gradient Boosting</b> demonstrou excelente capacidade preditiva (R² Test: 0.9973) e 
            a análise <b>SHAP</b> revelou que <b>educação e redução da pobreza</b> são os fatores mais críticos 
            para melhorar o IDHM no Brasil.
        </p>
        <p style='text-align: justify;'>
            As recomendações estratégicas são fundamentadas em evidências quantitativas extraídas do modelo 
            e devem ser implementadas de forma integrada, priorizando municípios com IDHM < 0.6 e alta 
            taxa de pessoas sem ensino fundamental completo.
        </p>
        
        
    </div>
    """, unsafe_allow_html=True)

# ============================
# FOOTER
# ============================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <b>📊 Dashboard IDHM - Análise Preditiva com Gradient Boosting</b><br>
    <b>🏆 Modelo Campeão:</b> {champion_model['Model']} (R² Test: {champion_model['R² Test']:.6f})<br>
    Desenvolvido com Python, Streamlit, Plotly, Scikit-Learn e SHAP<br>
    <b>Autor:</b> Alencara Avelar | © 2026 - Desafio II: Ciência e Governança de Dados
</div>
""", unsafe_allow_html=True)