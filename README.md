# ============================================
# 1. Importação de bibliotecas
# ============================================
import pandas as pd  # type: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore[reportMissingImports]
import importlib

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError:
    plt = None

from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[reportMissingModuleSource]
from xgboost import XGBClassifier  # type: ignore[reportMissingModuleSource]

from sklearn.metrics import classification_report, roc_curve, precision_recall_curve  # type: ignore[reportMissingModuleSource]
from imblearn.over_sampling import SMOTE  # type: ignore[reportMissingModuleSource]

import shap  # type: ignore[reportMissingModuleSource]

# ============================================
# 2. Carregamento e exploração dos dados
# ============================================
# Substitua pelo link do dataset fornecido na Aula 1
url = "LINK_DO_DATASET.csv"
df = pd.read_csv(url)

print("Primeiras linhas do dataset:")
print(df.head())

# Proporção de classes
fraude_ratio = df['Class'].value_counts(normalize=True)
print("Proporção de classes:\n", fraude_ratio)

# ============================================
# 3. Preparação dos dados
# ============================================
# Criar variável log do Amount
df['log_amount'] = np.log1p(df['Amount'])

# Separar features e target
X = df.drop(columns=['Class', 'Amount', 'Time'])
y = df['Class']

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================
# 4. Treinamento de modelos
# ============================================
# Baseline: Regressão Logística
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
log_reg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]))
xgb.fit(X_train, y_train)

# ============================================
# 5. Função de avaliação
# ============================================
def avaliar_modelo(modelo, X_test, y_test, nome):
    y_pred = modelo.predict(X_test)
    print(f"=== {nome} ===")
    print(classification_report(y_test, y_pred, digits=4))

    if plt is None:
        print("Matplotlib não está instalado; gráficos ignorados.")
        return

    # Curva ROC
    y_prob = modelo.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=nome)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Curva Precisão-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=nome)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

# Avaliar modelos
avaliar_modelo(log_reg, X_test, y_test, "Logistic Regression")
avaliar_modelo(rf, X_test, y_test, "Random Forest")
avaliar_modelo(xgb, X_test, y_test, "XGBoost")

# ============================================
# 6. Balanceamento com Oversampling (SMOTE)
# ============================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

log_reg_smote = LogisticRegression(max_iter=1000)
log_reg_smote.fit(X_res, y_res)
avaliar_modelo(log_reg_smote, X_test, y_test, "Logistic Regression + SMOTE")

# ============================================
# 7. Explicação com SHAP
# ============================================
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test[:100])  # exemplo com 100 amostras

# Importância global e explicação individual
if plt is not None:
    shap.summary_plot(shap_values, X_test[:100])
    shap.plots.waterfall(shap_values[0])
