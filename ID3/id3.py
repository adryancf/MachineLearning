import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from collections import Counter
from graphviz import Digraph
import warnings
warnings.filterwarnings('ignore')

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature    # Atributo usado para divisão
        self.threshold = threshold  # Limite de quartil
        self.value = value        # Valor da classe (para nós folha)
        self.left = left          # Subárvore esquerda (<= quartil)
        self.right = right        # Subárvore direita (> quartil)

class ID3Classifier:
    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None
        
    def _entropy(self, y):
        counts = Counter(y)
        entropy = 0.0
        total = len(y)
        for cls in counts:
            p = counts[cls] / total
            entropy -= p * np.log2(p)
        return entropy
    
    def _information_gain(self, X_col, y, threshold):
        left_idx = X_col <= threshold
        right_idx = X_col > threshold
        
        n = len(y)
        n_left, n_right = sum(left_idx), sum(right_idx)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        e_left = self._entropy(y[left_idx])
        e_right = self._entropy(y[right_idx])
        
        return self._entropy(y) - (n_left/n * e_left + n_right/n * e_right)
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                gain = self._information_gain(X[feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        # Critérios de parada (1 - Se todas as classes forem iguais ou 2 - Se não houver mais atributos)
        # Critério 1: Se não há amostras (NOVO - evita o erro)
        if len(y) == 0:
            return Node(value='Normal')  # Classe padrão

        if len(set(y)) == 1:
            return Node(value=y.iloc[0])
    
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Selecionar melhor divisão
        feature, threshold = self._best_split(X, y)
        
        # Divisão dos dados
        left_idx = X[feature] <= threshold
        right_idx = X[feature] > threshold
        
        # Construir subárvores
        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
        
    def predict(self, X):
        return [self._predict_sample(x, self.tree) for _, x in X.iterrows()]
    
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
            
        if node.value is not None:
            print(f"{indent}Class: {node.value}")
            return
        
        print(f"{indent}{node.feature} <= {node.threshold:.2f}?")
        print(f"{indent}--> True:")
        self.print_tree(node.left, indent + "    ")
        print(f"{indent}--> False:")
        self.print_tree(node.right, indent + "    ")
        
    def plot_tree_graphviz(self, filename='arvore_id3'):
        """Cria visualização gráfica da árvore usando Graphviz"""
        dot = Digraph(comment='Árvore ID3')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled')
        
        def add_nodes(node, parent_name=None, edge_label=None, node_id=0):
            current_name = f'node_{node_id}'
            
            if node.value is not None:
                # Nó folha
                label = f'CLASSE\n{node.value}'
                dot.node(current_name, label, fillcolor='lightgreen', fontsize='12', fontweight='bold')
            else:
                # Nó interno
                label = f'{node.feature}\n≤ {node.threshold:.2f}?'
                dot.node(current_name, label, fillcolor='lightblue', fontsize='11', fontweight='bold')
                
                # Adicionar filhos
                left_id = node_id * 2 + 1
                right_id = node_id * 2 + 2
                
                add_nodes(node.left, current_name, 'SIM', left_id)
                add_nodes(node.right, current_name, 'NÃO', right_id)
            
            # Conectar ao pai
            if parent_name:
                dot.edge(parent_name, current_name, label=edge_label, fontsize='10', fontweight='bold')
            
            return current_name
        
        add_nodes(self.tree)
        
        # Salvar e renderizar
        dot.render(filename, format='png', cleanup=True)
        print(f"Árvore salva como '{filename}.png'")
        return dot

# Função de pré-processamento
def preprocess_data(file_path, random_state=42):
    # Carregar dados
    df = pd.read_csv(file_path)
    
    # Tratamento de outliers usando Z-score (|Z| > 3) | Eliminação de linhas com valores extremos
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Discretização por quartis | Transformação de variáveis numéricas em categóricas [Contínuas para discretas]
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
    
    # Divisão estratificada [Treino e Teste]
    X = df.iloc[:, :-1] # Seleciona todas as colunas menos a última como características
    y = df.iloc[:, -1] # Seleciona a última coluna como rótulo

    # Divisão dos dados em treino e teste
    # 70% do arquivo para treino e 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
