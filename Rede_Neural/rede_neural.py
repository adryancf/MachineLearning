import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import tensorflow as tf
import time
import warnings
warnings.filterwarnings('ignore')

# Configuração para reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ConfiguracaoRedeNeural:
    """Configurações centralizadas para a Rede Neural."""
    
    # ARQUITETURA DO MODELO
    HIDDEN_LAYERS = [32, 16]           # Camadas ocultas e neurônios
    DROPOUT_RATE = 0.3                # Taxa de dropout
    L2_REGULARIZATION = 0.001          # Regularização L2
    
    # OTIMIZAÇÃO
    LEARNING_RATE = 0.001              # Taxa de aprendizado
    OPTIMIZER = 'adam'                 # Otimizador
    
    # TREINAMENTO
    EPOCHS = 150                       # Máximo de épocas
    BATCH_SIZE = 32                    # Tamanho do lote
    
    # EARLY STOPPING
    EARLY_STOPPING_PATIENCE = 20       # Épocas sem melhoria para parar
    
    # PRÉ-PROCESSAMENTO
    NORMALIZACAO = 'standard'          # Tipo de normalização
    
    # DIVISÃO DOS DADOS
    TEST_SIZE = 0.3                    # Proporção para teste
    VALIDATION_SIZE = 0.2              # Proporção para validação
    
    # VALIDAÇÃO CRUZADA
    CV_FOLDS = 5                       # Número de folds
    CV_EPOCHS = 50                     # Épocas para cada fold
    
    # IMPORTÂNCIA DAS FEATURES
    PERMUTATION_REPEATS = 10           # Repetições para cálculo de importância
    
    # TESTE DE ROBUSTEZ
    NOISE_LEVELS = [0.1, 0.2, 0.3]    # Níveis de ruído gaussiano
    
    @classmethod
    def imprimir_configuracoes(cls):
        """Imprime todas as configurações atuais."""
        print("\n" + "="*70)
        print("CONFIGURAÇÕES ATUAIS DA REDE NEURAL")
        print("="*70)
        
        print(f"\n🏗️  ARQUITETURA:")
        print(f"   Camadas Ocultas: {cls.HIDDEN_LAYERS}")
        print(f"   Taxa de Dropout: {cls.DROPOUT_RATE}")
        print(f"   Regularização L2: {cls.L2_REGULARIZATION}")
        
        print(f"\n⚙️  OTIMIZAÇÃO:")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Otimizador: {cls.OPTIMIZER}")
        
        print(f"\n🎯 TREINAMENTO:")
        print(f"   Épocas Máximas: {cls.EPOCHS}")
        print(f"   Tamanho do Lote: {cls.BATCH_SIZE}")
        print(f"   Early Stopping: {cls.EARLY_STOPPING_PATIENCE} épocas")
        
        print("="*70)

class FuncoesAtivacao:
    """Funções de ativação implementadas."""
    
    @staticmethod
    def relu(x):
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivada(x):
        """Derivada da ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        """Softmax estável numericamente"""
        # Subtrair o máximo para estabilidade numérica
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivada(x):
        """Derivada do softmax (não usada diretamente no backprop)"""
        s = FuncoesAtivacao.softmax(x)
        return s * (1 - s)

class RedeNeural:
    """Implementação de rede neural com otimizadores do TensorFlow."""
    
    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.config = ConfiguracaoRedeNeural()
        self.weights = []
        self.biases = []
        self.tf_weights = []  # Variáveis TensorFlow
        self.tf_biases = []   # Variáveis TensorFlow
        self.optimizer = None
        self.scaler = None
        self.feature_names = None
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Configurar seeds
        np.random.seed(random_state)
        
    def inicializar_pesos(self, arquitetura):
        """Inicializa pesos e biases usando Xavier/Glorot initialization."""
        self.weights = []
        self.biases = []
        self.tf_weights = []
        self.tf_biases = []
        
        for i in range(len(arquitetura) - 1):
            # Xavier/Glorot initialization
            fan_in = arquitetura[i]
            fan_out = arquitetura[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            # Inicializar pesos
            W = np.random.uniform(-limit, limit, (arquitetura[i], arquitetura[i + 1])).astype(np.float32)
            self.weights.append(W)
            
            # Inicializar biases com zeros
            b = np.zeros((1, arquitetura[i + 1]), dtype=np.float32)
            self.biases.append(b)
            
            # Criar variáveis TensorFlow
            w_tf = tf.Variable(W, dtype=tf.float32, name=f'weight_{i}')
            b_tf = tf.Variable(b, dtype=tf.float32, name=f'bias_{i}')
            
            self.tf_weights.append(w_tf)
            self.tf_biases.append(b_tf)
        
        print(f"✓ Pesos inicializados para arquitetura: {arquitetura}")
    
    def inicializar_otimizador(self):
        """Inicializa um único otimizador para todas as variáveis."""
        # Coletar todas as variáveis
        all_variables = self.tf_weights + self.tf_biases
        
        # Criar otimizador
        if self.config.OPTIMIZER == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError("Otimizador deve ser 'adam', 'rmsprop' ou 'sgd'")
        
        # "Construir" o otimizador fazendo uma chamada dummy
        dummy_grads = [tf.zeros_like(var) for var in all_variables]
        self.optimizer.apply_gradients(zip(dummy_grads, all_variables))
        
        print(f"✓ Otimizador {self.config.OPTIMIZER} inicializado para {len(all_variables)} variáveis")
    
    def sincronizar_pesos_numpy(self):
        """Sincroniza os pesos numpy com as variáveis TensorFlow."""
        for i in range(len(self.weights)):
            self.weights[i] = self.tf_weights[i].numpy()
            self.biases[i] = self.tf_biases[i].numpy()
    
    def sincronizar_pesos_tf(self):
        """Sincroniza as variáveis TensorFlow com os pesos numpy."""
        for i in range(len(self.weights)):
            self.tf_weights[i].assign(self.weights[i])
            self.tf_biases[i].assign(self.biases[i])
    
    def forward_propagation(self, X, training=True):
        """Forward propagation."""
        activations = [X]
        z_values = []  # Valores antes da ativação (para backprop)
        
        # Através das camadas ocultas
        for i in range(len(self.weights) - 1):
            # Calcular z = X * W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Aplicar ReLU
            a = FuncoesAtivacao.relu(z)
            
            # Aplicar dropout apenas durante treinamento
            if training and self.config.DROPOUT_RATE > 0:
                dropout_mask = np.random.binomial(1, 1 - self.config.DROPOUT_RATE, a.shape) / (1 - self.config.DROPOUT_RATE)
                a = a * dropout_mask
            
            activations.append(a)
        
        # Camada de saída (softmax)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        
        # Aplicar softmax
        output = FuncoesAtivacao.softmax(z_output)
        activations.append(output)
        
        return activations, z_values
    
    def calcular_loss(self, y_true, y_pred):
        """Calcula cross-entropy loss com regularização L2."""
        # Cross-entropy loss
        # Evitar log(0) adicionando pequeno epsilon
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Converter y_true para one-hot se necessário
        if len(y_true.shape) == 1:
            y_true_one_hot = np.eye(y_pred.shape[1])[y_true]
        else:
            y_true_one_hot = y_true
        
        cross_entropy = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_clipped), axis=1))
        
        # Regularização L2
        l2_reg = 0
        for W in self.weights:
            l2_reg += np.sum(W ** 2)
        l2_reg *= self.config.L2_REGULARIZATION / 2
        
        total_loss = cross_entropy + l2_reg
        return total_loss, cross_entropy
    
    def backward_propagation(self, X, y_true, activations, z_values):
        """Backward propagation."""
        m = X.shape[0]  # Número de amostras
        gradients_w = []
        gradients_b = []
        
        # Converter y_true para one-hot
        if len(y_true.shape) == 1:
            y_true_one_hot = np.eye(activations[-1].shape[1])[y_true]
        else:
            y_true_one_hot = y_true
        
        # Gradiente da camada de saída (softmax + cross-entropy)
        dz = activations[-1] - y_true_one_hot  # Derivada simplificada de softmax + cross-entropy
        
        # Backward através de todas as camadas
        for i in reversed(range(len(self.weights))):
            # Gradientes dos pesos e biases
            if i == len(self.weights) - 1:
                # Última camada
                dW = np.dot(activations[i].T, dz) / m
                db = np.mean(dz, axis=0, keepdims=True)
            else:
                # Camadas ocultas
                dW = np.dot(activations[i].T, dz) / m
                db = np.mean(dz, axis=0, keepdims=True)
            
            # Adicionar regularização L2 aos gradientes dos pesos
            dW += self.config.L2_REGULARIZATION * self.weights[i]
            
            gradients_w.insert(0, dW.astype(np.float32))
            gradients_b.insert(0, db.astype(np.float32))
            
            # Calcular gradiente para a próxima camada (se não for a primeira)
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * FuncoesAtivacao.relu_derivada(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def atualizar_pesos(self, gradients_w, gradients_b):
        """Atualiza pesos usando otimizador do TensorFlow."""
        # Criar lista de gradientes na ordem correta
        gradients = []
        variables = []
        
        # Adicionar gradientes dos pesos
        for i in range(len(self.weights)):
            gradients.append(tf.constant(gradients_w[i], dtype=tf.float32))
            variables.append(self.tf_weights[i])
        
        # Adicionar gradientes dos biases
        for i in range(len(self.biases)):
            gradients.append(tf.constant(gradients_b[i], dtype=tf.float32))
            variables.append(self.tf_biases[i])
        
        # Aplicar gradientes
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        # Sincronizar com arrays numpy
        self.sincronizar_pesos_numpy()
    
    def treinar_batch(self, X_batch, y_batch):
        """Treina um batch."""
        # Forward propagation
        activations, z_values = self.forward_propagation(X_batch, training=True)
        
        # Calcular loss
        loss, cross_entropy = self.calcular_loss(y_batch, activations[-1])
        
        # Backward propagation
        gradients_w, gradients_b = self.backward_propagation(X_batch, y_batch, activations, z_values)
        
        # Atualizar pesos
        self.atualizar_pesos(gradients_w, gradients_b)
        
        # Calcular acurácia
        predictions = np.argmax(activations[-1], axis=1)
        if len(y_batch.shape) == 1:
            accuracy = np.mean(predictions == y_batch)
        else:
            accuracy = np.mean(predictions == np.argmax(y_batch, axis=1))
        
        return loss, accuracy
    
    def avaliar(self, X, y):
        """Avalia o modelo em um conjunto de dados."""
        activations, _ = self.forward_propagation(X, training=False)
        loss, _ = self.calcular_loss(y, activations[-1])
        
        predictions = np.argmax(activations[-1], axis=1)
        if len(y.shape) == 1:
            accuracy = np.mean(predictions == y)
        else:
            accuracy = np.mean(predictions == np.argmax(y, axis=1))
        
        return loss, accuracy, predictions
    
    def criar_batches(self, X, y, batch_size):
        """Cria batches aleatórios."""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def treinar(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """Treina a rede neural."""
        if epochs is None:
            epochs = self.config.EPOCHS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        print("Iniciando treinamento...")
        
        # Determinar arquitetura
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        arquitetura = [input_dim] + self.config.HIDDEN_LAYERS + [output_dim]
        
        # Inicializar pesos e otimizador
        self.inicializar_pesos(arquitetura)
        self.inicializar_otimizador()
        
        # Variáveis para early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        # Loop de treinamento
        for epoch in range(epochs):
            # Treino
            epoch_losses = []
            epoch_accuracies = []
            
            for X_batch, y_batch in self.criar_batches(X_train, y_train, batch_size):
                loss, accuracy = self.treinar_batch(X_batch, y_batch)
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
            
            # Médias da época
            train_loss = np.mean(epoch_losses)
            train_accuracy = np.mean(epoch_accuracies)
            
            # Validação
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy, _ = self.avaliar(X_val, y_val)
                
                # Salvar histórico
                self.training_history['loss'].append(train_loss)
                self.training_history['accuracy'].append(train_accuracy)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Salvar melhores pesos
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                # Imprimir progresso
                if epoch % 10 == 0 or patience_counter == 0:
                    print(f"Época {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, "
                          f"Val_Loss: {val_loss:.4f}, Val_Acc: {val_accuracy:.4f}")
                
                # Parar se não melhorar
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping na época {epoch+1}")
                    # Restaurar melhores pesos
                    if best_weights is not None and best_biases is not None:
                        self.weights = best_weights
                        self.biases = best_biases
                        self.sincronizar_pesos_tf()
                    break
            else:
                # Sem validação
                self.training_history['loss'].append(train_loss)
                self.training_history['accuracy'].append(train_accuracy)
                
                if epoch % 10 == 0:
                    print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")
        
        print("✓ Treinamento concluído!")
    
    def predict(self, X):
        """Faz predições."""
        activations, _ = self.forward_propagation(X, training=False)
        return activations[-1]
    
    def predict_classes(self, X):
        """Faz predições de classes."""
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

class RedeNeuralClassificador:
    """Wrapper para compatibilidade com código existente."""
    
    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.model = RedeNeural(random_state)
        self.scaler = None
        self.feature_names = None
        self.config = ConfiguracaoRedeNeural()
        
    def validar_schema(self, data):
        """Valida o schema dos dados de entrada."""
        print("Validando schema dos dados...")
        
        # Verificar se há 8 colunas (1 ID + 6 features + 1 target)
        if data.shape[1] != 8:
            raise ValueError(f"Esperado 8 colunas (ID + 6 features + target), encontrado {data.shape[1]}")
        
        # Remover a primeira coluna (ID) se existir
        if data.columns[0].lower() in ['id', 'index', '0'] or data.iloc[:, 0].dtype == 'int64':
            print("Removendo coluna de ID...")
            data = data.drop(data.columns[0], axis=1)
        
        # Agora devemos ter 7 colunas (6 features + 1 target)
        if data.shape[1] != 7:
            raise ValueError(f"Após remoção do ID, esperado 7 colunas, encontrado {data.shape[1]}")
        
        # Verificar tipos de dados das features (primeiras 6 colunas)
        for i in range(6):
            if not pd.api.types.is_numeric_dtype(data.iloc[:, i]):
                print(f"Aviso: Coluna {i+1} não é numérica, tentando conversão...")
                data.iloc[:, i] = pd.to_numeric(data.iloc[:, i], errors='coerce')
        
        # Verificar valores da classe (última coluna)
        classes_validas = [1, 2, 3, 4]
        classes_unicas = data.iloc[:, -1].unique()
        for classe in classes_unicas:
            if classe not in classes_validas:
                raise ValueError(f"Classe inválida encontrada: {classe}. Classes válidas: {classes_validas}")
        
        print("✓ Schema validado com sucesso!")
        print(f"Dados após validação: {data.shape}")
        return data
    
    def tratar_valores_ausentes(self, data):
        """Trata valores ausentes ou inválidos."""
        print("Tratando valores ausentes...")
        
        # Estatísticas antes do tratamento
        missing_before = data.isnull().sum().sum()
        print(f"Valores ausentes encontrados: {missing_before}")
        
        if missing_before > 0:
            # Para features numéricas: usar mediana
            for i in range(6):
                if data.iloc[:, i].isnull().any():
                    mediana = data.iloc[:, i].median()
                    data.iloc[:, i].fillna(mediana, inplace=True)
                    print(f"Coluna {i}: preenchida com mediana = {mediana:.2f}")
            
            # Para target: remover linhas com valores ausentes
            data = data.dropna(subset=[data.columns[-1]])
        
        print("✓ Valores ausentes tratados!")
        return data
    
    def preprocessar_dados(self, data, metodo_normalizacao=None):
        """Pré-processa os dados."""
        if metodo_normalizacao is None:
            metodo_normalizacao = self.config.NORMALIZACAO
            
        print(f"Iniciando pré-processamento (normalização: {metodo_normalizacao})...")
        
        # Separar features e target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Converter classes para índices (1,2,3,4 -> 0,1,2,3)
        y = y - 1
        
        # Normalização das features
        if metodo_normalizacao == 'standard':
            self.scaler = StandardScaler()
        elif metodo_normalizacao == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Método deve ser 'standard' ou 'minmax'")
        
        X_scaled = self.scaler.fit_transform(X)
        
        print("✓ Pré-processamento concluído!")
        return X_scaled, y
    
    def executar_treinamento(self, data_path, feature_names=None):
        """Executa o treinamento completo."""
        import time
        tempo_inicio = time.time()
        
        print("="*60)
        print("INICIANDO TREINAMENTO DA REDE NEURAL")
        print("="*60)
        
        # Imprimir configurações atuais
        self.config.imprimir_configuracoes()
        
        # 1. Carregar e validar dados
        print("\n1. CARREGAMENTO E VALIDAÇÃO DOS DADOS")
        print("-"*40)
        
        data = pd.read_csv(data_path, header=None)
        print(f"Dados carregados: {data.shape}")
        
        # Nomear as colunas
        data.columns = ['ID', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Target']
        
        data = self.validar_schema(data)
        data = self.tratar_valores_ausentes(data)
        
        if feature_names is None:
            feature_names = ['Sinal Vital 1', 'Sinal Vital 2', 'Sinal Vital 3', 'Sinal Vital 4', 'Sinal Vital 5', 'Sinal Vital 6']
        self.feature_names = feature_names
        
        # 2. Análise exploratória básica
        print("\n1.1. ANÁLISE EXPLORATÓRIA")
        print("-"*40)
        print("Distribuição das classes:")
        print(data.iloc[:, -1].value_counts().sort_index())
        
        # 3. Pré-processamento
        print("\n2. PRÉ-PROCESSAMENTO")
        print("-"*40)
        X, y = self.preprocessar_dados(data)
        
        # 4. Divisão estratificada
        print("\n3. DIVISÃO DOS DADOS")
        print("-"*40)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, stratify=y, random_state=self.random_state
        )
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=self.config.VALIDATION_SIZE, stratify=y_train, random_state=self.random_state
        )
        
        print(f"Treino: {X_train_split.shape[0]} amostras")
        print(f"Validação: {X_val.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        
        # 5. Treinar modelo
        print("\n4. TREINAMENTO DO MODELO")
        print("-"*40)
        tempo_treinamento_inicio = time.time()
        
        self.model.treinar(X_train_split, y_train_split, X_val, y_val)
        
        tempo_treinamento = time.time() - tempo_treinamento_inicio
        print(f"⏱️  Tempo de treinamento: {tempo_treinamento:.2f}s")
        
        # 6. Avaliação final
        print("\n5. AVALIAÇÃO FINAL")
        print("-"*40)
        y_pred = self.model.predict_classes(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Acurácia: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']))
        
        tempo_total = time.time() - tempo_inicio
        print(f"\n⏱️  Tempo total: {tempo_total:.2f}s ({tempo_total/60:.1f} min)")
        
        return {
            'X_train': X_train_split,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_split,
            'y_val': y_val,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'f1_score': f1,
            'tempo_total': tempo_total
        }
    
    def calcular_importancia_features(self, X_test, y_test):
        """Calcula a importância das features usando permutation importance."""
        print("Calculando importância das features...")
        
        # Acurácia base (sem permutação)
        y_pred_base = self.model.predict_classes(X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        print(f"Acurácia base: {acc_base:.4f}")
        
        # Calcular importância para cada feature
        n_features = X_test.shape[1]
        importancias = []
        importancias_std = []
        
        for i in range(n_features):
            print(f"Processando feature {i+1}/{n_features}...")
            
            # Lista para armazenar as diferenças de acurácia
            diferencas = []
            
            # Repetir várias vezes para ter estatística mais robusta
            for rep in range(self.config.PERMUTATION_REPEATS):
                # Copiar os dados
                X_perm = X_test.copy()
                
                # Permutar a feature i
                np.random.seed(self.random_state + rep)
                X_perm[:, i] = np.random.permutation(X_perm[:, i])
                
                # Calcular nova acurácia
                y_pred_perm = self.model.predict_classes(X_perm)
                acc_perm = accuracy_score(y_test, y_pred_perm)
                
                # Diferença (importância = queda na acurácia)
                diferencas.append(acc_base - acc_perm)
            
            # Estatísticas da importância
            media_imp = np.mean(diferencas)
            std_imp = np.std(diferencas)
            
            importancias.append(media_imp)
            importancias_std.append(std_imp)
            
            print(f"Feature {i+1}: Importância = {media_imp:.4f} (±{std_imp:.4f})")
        
        # Criar objeto similar ao sklearn
        class PermutationImportanceResult:
            def __init__(self, importances_mean, importances_std):
                self.importances_mean = np.array(importances_mean)
                self.importances_std = np.array(importances_std)
        
        print("✓ Cálculo de importância concluído!")
        return PermutationImportanceResult(importancias, importancias_std)
    
    def teste_robustez(self, X_test, y_test, niveis_ruido=None):
        """Testa robustez adicionando ruído gaussiano."""
        if niveis_ruido is None:
            niveis_ruido = self.config.NOISE_LEVELS
            
        print("Testando robustez com ruído gaussiano...")
        
        resultados = []
        
        # Teste sem ruído
        y_pred_sem_ruido = self.model.predict_classes(X_test)
        acc_sem_ruido = accuracy_score(y_test, y_pred_sem_ruido)
        resultados.append(('Sem ruído', 0.0, acc_sem_ruido))
        print(f"Acurácia sem ruído: {acc_sem_ruido:.4f}")
        
        # Testes com ruído
        for nivel in niveis_ruido:
            ruido = np.random.normal(0, nivel, X_test.shape)
            X_test_ruido = X_test + ruido
            
            y_pred_ruido = self.model.predict_classes(X_test_ruido)
            acc_ruido = accuracy_score(y_test, y_pred_ruido)
            resultados.append((f'Ruído σ={nivel}', nivel, acc_ruido))
            print(f"Acurácia com ruído σ={nivel}: {acc_ruido:.4f}")
        
        return resultados

