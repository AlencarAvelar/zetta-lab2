# DESAFIO II - Comparação de Modelos de Machine Learning
# ========================================================
# Script para comparação de múltiplos modelos preditivos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==========================
# 1. CARREGAMENTO DOS DADOS
# ==========================

print("Carregando dados...")
df = pd.read_csv('/home/alencaravelar/Desktop/zetta-lab/zetta-lab2/zetta-lab2/data/refined/base_udh_refined.csv')

# Definir features e target
X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dados carregados: {X.shape[0]} observações, {X.shape[1]} features")
print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}\n")

# ================================
# 2. DEFINIÇÃO DOS MODELOS
# ================================

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=100),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.01)
}

# =================================
# 3. TREINAMENTO E AVALIAÇÃO
# =================================

results = []

print("="*80)
print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print("="*80 + "\n")

for name, model in models.items():
    print(f"Treinando {name}...")
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'Model': name,
        'R² Train': r2_train,
        'R² Test': r2_test,
        'MAE': mae,
        'RMSE': rmse,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std,
        'Overfitting': r2_train - r2_test
    })
    
    print(f"  R² Test: {r2_test:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}\n")

# ====================================
# 4. CRIAR DATAFRAME DE RESULTADOS
# ====================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² Test', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("RESULTADOS COMPARATIVOS (Ordenados por R² Test)")
print("="*80)
print(results_df.to_string(index=False))

# Salvar resultados
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✅ Resultados salvos em 'model_comparison_results.csv'")

# ====================================
# 5. VISUALIZAÇÕES
# ====================================

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# 5.1 Comparação de R² Test
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: R² Test
ax1 = axes[0, 0]
colors = sns.color_palette('viridis', len(results_df))
results_df.plot(kind='barh', x='Model', y='R² Test', ax=ax1, color=colors, legend=False)
ax1.set_xlabel('R² Test', fontsize=12, fontweight='bold')
ax1.set_ylabel('Modelo', fontsize=12, fontweight='bold')
ax1.set_title('Comparação: R² Test', fontsize=14, fontweight='bold')
ax1.axvline(x=results_df['R² Test'].mean(), color='red', linestyle='--', label='Média')
ax1.legend()

# Gráfico 2: MAE
ax2 = axes[0, 1]
results_df.plot(kind='barh', x='Model', y='MAE', ax=ax2, color=colors, legend=False)
ax2.set_xlabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax2.set_ylabel('')
ax2.set_title('Comparação: MAE (menor é melhor)', fontsize=14, fontweight='bold')

# Gráfico 3: RMSE
ax3 = axes[1, 0]
results_df.plot(kind='barh', x='Model', y='RMSE', ax=ax3, color=colors, legend=False)
ax3.set_xlabel('Root Mean Squared Error', fontsize=12, fontweight='bold')
ax3.set_ylabel('Modelo', fontsize=12, fontweight='bold')
ax3.set_title('Comparação: RMSE (menor é melhor)', fontsize=14, fontweight='bold')

# Gráfico 4: Overfitting (diferença R² Train - Test)
ax4 = axes[1, 1]
results_df.plot(kind='barh', x='Model', y='Overfitting', ax=ax4, color=colors, legend=False)
ax4.set_xlabel('Overfitting (R² Train - R² Test)', fontsize=12, fontweight='bold')
ax4.set_ylabel('')
ax4.set_title('Análise de Overfitting (menor é melhor)', fontsize=14, fontweight='bold')
ax4.axvline(x=0.01, color='orange', linestyle='--', label='Threshold 0.01')
ax4.legend()

plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico de comparação salvo em 'model_comparison_metrics.png'")
plt.close()

# 5.2 Visualização de R² Train vs Test
plt.figure(figsize=(12, 8))
x = np.arange(len(results_df))
width = 0.35

plt.barh(x - width/2, results_df['R² Train'], width, label='R² Train', alpha=0.8)
plt.barh(x + width/2, results_df['R² Test'], width, label='R² Test', alpha=0.8)

plt.ylabel('Modelo', fontweight='bold')
plt.xlabel('R² Score', fontweight='bold')
plt.title('R² Train vs R² Test - Análise de Generalização', fontsize=14, fontweight='bold')
plt.yticks(x, results_df['Model'])
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_r2_train_vs_test.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico R² Train vs Test salvo em 'model_r2_train_vs_test.png'")
plt.close()

# ====================================
# 6. ANÁLISE E RECOMENDAÇÕES
# ====================================

print("\n" + "="*80)
print("ANÁLISE E RECOMENDAÇÕES")
print("="*80)

best_model = results_df.iloc[0]
print(f"\n🏆 MELHOR MODELO: {best_model['Model']}")
print(f"   - R² Test: {best_model['R² Test']:.6f}")
print(f"   - MAE: {best_model['MAE']:.6f}")
print(f"   - RMSE: {best_model['RMSE']:.6f}")
print(f"   - Overfitting: {best_model['Overfitting']:.6f}")

print("\n📊 TOP 3 MODELOS:")
for i, row in results_df.head(3).iterrows():
    print(f"{i+1}. {row['Model']} - R² Test: {row['R² Test']:.6f}")

print("\n⚠️  MODELOS COM OVERFITTING SIGNIFICATIVO (>0.05):")
overfitting_models = results_df[results_df['Overfitting'] > 0.05]
if len(overfitting_models) > 0:
    for _, row in overfitting_models.iterrows():
        print(f"   - {row['Model']}: {row['Overfitting']:.6f}")
else:
    print("   ✅ Nenhum modelo apresentou overfitting significativo")

print("\n" + "="*80)
print("SCRIPT FINALIZADO COM SUCESSO!")
print("="*80)
