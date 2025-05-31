import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from collections import Counter
from graphviz import Digraph
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature    # Atributo usado para divisão
        self.threshold = threshold  # Limite de quartil
        self.value = value        # Valor da classe (para nós folha)
        self.left = left          # Subárvore esquerda (<= quartil)
        self.right = right        # Subárvore direita (> quartil)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.feature_importances_ = {}
        
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        entropy = 0.0
        total = len(y)
        for cls in counts:
            p = counts[cls] / total
            if p > 0:
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
    
    def _best_split(self, X, y, available_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in available_features:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                gain = self._information_gain(X[feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0, available_features=None):
        # Inicializar features disponíveis na primeira chamada
        if available_features is None:
            if self.max_features is None:
                available_features = list(X.columns)
            else:
                np.random.seed(self.random_state + depth)
                available_features = np.random.choice(
                    list(X.columns), 
                    size=min(self.max_features, len(X.columns)), 
                    replace=False
                ).tolist()
        
        # Critérios de parada
        if len(y) == 0:
            return Node(value='1')  # Classe padrão
        
        # Verificar se há amostras suficientes
        if len(y) < self.min_samples_leaf:
            return Node(value=Counter(y).most_common(1)[0][0])
            
        # Verificar se todas as amostras pertencem à mesma classe
        if len(set(y)) == 1:
            return Node(value=y.iloc[0])
    
        # Verificar profundidade máxima
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Verificar se há features disponíveis
        if len(available_features) == 0:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Selecionar melhor divisão
        feature, threshold, gain = self._best_split(X, y, available_features)
        
        if feature is None or gain == 0:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Registrar importância do atributo
        if feature in self.feature_importances_:
            self.feature_importances_[feature] += gain
        else:
            self.feature_importances_[feature] = gain
        
        # Divisão dos dados
        left_idx = X[feature] <= threshold
        right_idx = X[feature] > threshold
        
        # Verificar se a divisão é válida
        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Construir subárvores
        left = self._build_tree(X[left_idx], y[left_idx], depth+1, available_features)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1, available_features)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.feature_importances_ = {}
        self.tree = self._build_tree(X, y)
        
        # Normalizar importâncias
        total_importance = sum(self.feature_importances_.values())
        if total_importance > 0:
            self.feature_importances_ = {
                k: v/total_importance for k, v in self.feature_importances_.items()
            }
        
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
        
    def predict(self, X):
        if self.tree is None:
            return ['1'] * len(X)
        return [self._predict_sample(x, self.tree) for _, x in X.iterrows()]

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, 
                 max_features='sqrt', bootstrap=True, random_state=42, verbose=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []
        self.feature_importances_ = {}
        self.training_time_ = 0
        self.tree_times_ = []
        
    def _bootstrap_sample(self, X, y, random_state):
        np.random.seed(random_state)
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]
    
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """Treina a Random Forest com medição de tempo e barra de progresso"""
        
        # Inicializar cronômetro
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🌲 INICIANDO TREINAMENTO RANDOM FOREST")
            print(f"{'='*60}")
            print(f"📊 Parâmetros:")
            print(f"   • Número de árvores: {self.n_estimators}")
            print(f"   • Max depth: {self.max_depth}")
            print(f"   • Min samples leaf: {self.min_samples_leaf}")
            print(f"   • Max features: {self.max_features}")
            print(f"   • Bootstrap: {self.bootstrap}")
            print(f"   • Dataset: {len(X)} amostras, {len(X.columns)} features")
            print(f"{'='*60}")
        
        self.trees = []
        self.feature_importances_ = {col: 0.0 for col in X.columns}
        self.tree_times_ = []
        
        max_features = self._get_max_features(len(X.columns))
        
        # Barra de progresso com tqdm
        if self.verbose:
            progress_bar = tqdm(range(self.n_estimators), 
                               desc="🌳 Treinando árvores", 
                               unit="árvore",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            progress_bar = range(self.n_estimators)
        
        for i in progress_bar:
            tree_start_time = time.time()
            
            # Criar árvore individual
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + i
            )
            
            # Amostragem bootstrap
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y, self.random_state + i)
            else:
                X_sample, y_sample = X, y
            
            # Treinar árvore
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # Acumular importâncias
            for feature, importance in tree.feature_importances_.items():
                self.feature_importances_[feature] += importance
            
            # Medir tempo da árvore
            tree_time = time.time() - tree_start_time
            self.tree_times_.append(tree_time)
            
            # Atualizar barra de progresso com informações detalhadas
            if self.verbose and hasattr(progress_bar, 'set_postfix'):
                avg_time = np.mean(self.tree_times_)
                remaining_trees = self.n_estimators - (i + 1)
                eta = avg_time * remaining_trees
                
                progress_bar.set_postfix({
                    'Tempo/árvore': f'{tree_time:.2f}s',
                    'Média': f'{avg_time:.2f}s',
                    'ETA': f'{eta:.1f}s'
                })
        
        # Normalizar importâncias finais
        total_importance = sum(self.feature_importances_.values())
        if total_importance > 0:
            self.feature_importances_ = {
                k: v/total_importance for k, v in self.feature_importances_.items()
            }
        
        # Calcular tempo total
        self.training_time_ = time.time() - start_time
        
        if self.verbose:
            print(f"\n✅ TREINAMENTO CONCLUÍDO!")
            print(f"{'='*60}")
            print(f"⏱️  Tempo total: {self.training_time_:.2f} segundos")
            print(f"⚡ Tempo médio por árvore: {np.mean(self.tree_times_):.3f}s")
            print(f"🚀 Árvore mais rápida: {min(self.tree_times_):.3f}s")
            print(f"🐌 Árvore mais lenta: {max(self.tree_times_):.3f}s")
            print(f"📊 Throughput: {self.n_estimators/self.training_time_:.1f} árvores/segundo")
            print(f"{'='*60}")
    
    def predict(self, X):
        if not self.trees:
            return ['1'] * len(X)
        
        # Coletar predições de todas as árvores
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_predictions.append(predictions)
        
        # Votação majoritária
        final_predictions = []
        for i in range(len(X)):
            votes = [pred[i] for pred in all_predictions]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        
        return final_predictions
    
    def plot_tree_graphviz(self, tree_idx=0, filename='arvore_random_forest'):
        """Visualiza uma árvore específica da floresta"""
        if tree_idx >= len(self.trees):
            print(f"Árvore {tree_idx} não existe. Máximo: {len(self.trees)-1}")
            return
        
        tree = self.trees[tree_idx]
        dot = Digraph(comment=f'Árvore {tree_idx} da Random Forest')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled')
        
        def add_nodes(node, parent_name=None, edge_label=None, node_id=0):
            current_name = f'node_{node_id}'
            
            if node.value is not None:
                label = f'CLASSE\n{node.value}'
                dot.node(current_name, label, fillcolor='lightgreen', fontsize='12', fontweight='bold')
            else:
                label = f'{node.feature}\n≤ {node.threshold:.2f}?'
                dot.node(current_name, label, fillcolor='lightblue', fontsize='11', fontweight='bold')
                
                left_id = node_id * 2 + 1
                right_id = node_id * 2 + 2
                
                add_nodes(node.left, current_name, 'SIM', left_id)
                add_nodes(node.right, current_name, 'NÃO', right_id)
            
            if parent_name:
                dot.edge(parent_name, current_name, label=edge_label, fontsize='10', fontweight='bold')
            
            return current_name
        
        add_nodes(tree.tree)
        
        filename_final = f"{filename}_tree_{tree_idx}"
        dot.render(filename_final, format='png', cleanup=True)
        print(f"Árvore {tree_idx} salva como '{filename_final}.png'")
        return dot

# Função de pré-processamento
def preprocess_data(file_path, random_state=42):
    df = pd.read_csv(file_path)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], q=8, labels=False, duplicates='drop')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(str)  # Converter para string para consistência

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
