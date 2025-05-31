import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    L2_REGULARIZATION = 0.001           # Regularização L2
    
    # OTIMIZAÇÃO
    LEARNING_RATE = 0.001              # Taxa de aprendizado
    OPTIMIZER = 'adam'                 # Otimizador
    
    # TREINAMENTO
    EPOCHS = 150                       # Máximo de épocas
    BATCH_SIZE = 32                    # Tamanho do lote
    
    # EARLY STOPPING
    EARLY_STOPPING_PATIENCE = 20       # Épocas sem melhoria para parar
    REDUCE_LR_PATIENCE = 10            # Épocas sem melhoria para reduzir LR
    REDUCE_LR_FACTOR = 0.5             # Fator de redução do LR
    MIN_LEARNING_RATE = 1e-6           # LR mínimo
    
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
        
        print(f"\n📊 DADOS:")
        print(f"   Normalização: {cls.NORMALIZACAO}")
        print(f"   Teste: {cls.TEST_SIZE*100:.0f}%")
        print(f"   Validação: {cls.VALIDATION_SIZE*100:.0f}%")
        print(f"   CV Folds: {cls.CV_FOLDS}")
        
        print("="*70)

class RedeNeuralClassificador:
    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.history = None
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
    
    def criar_modelo(self, input_dim, num_classes=4, hidden_layers=None, dropout_rate=None, l2_reg=None):
        """Cria o modelo da rede neural."""
        # Usar configurações padrão se não especificadas
        if hidden_layers is None:
            hidden_layers = self.config.HIDDEN_LAYERS
        if dropout_rate is None:
            dropout_rate = self.config.DROPOUT_RATE
        if l2_reg is None:
            l2_reg = self.config.L2_REGULARIZATION
            
        print(f"Criando modelo com arquitetura: {hidden_layers}")
        
        model = keras.Sequential()
        
        # Camada de entrada
        model.add(layers.Dense(hidden_layers[0], 
                              input_dim=input_dim,
                              activation='relu',
                              kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
        
        # Camadas ocultas
        for neurons in hidden_layers[1:]:
            model.add(layers.Dense(neurons, 
                                  activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(l2_reg)))
            model.add(layers.Dropout(dropout_rate))
        
        # Camada de saída
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Configurar otimizador
        if self.config.OPTIMIZER == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=self.config.LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError("Otimizador deve ser 'adam', 'rmsprop' ou 'sgd'")
        
        # Compilar modelo
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Modelo criado e compilado!")
        return model
    
    def treinar_modelo(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """Treina o modelo com early stopping e redução de learning rate."""
        if epochs is None:
            epochs = self.config.EPOCHS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        print("Iniciando treinamento...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LEARNING_RATE,
            verbose=1
        )
        
        # Treinamento
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✓ Treinamento concluído!")
        return self.history
    
    def validacao_cruzada(self, X, y, k=None):
        """Realiza validação cruzada estratificada."""
        if k is None:
            k = self.config.CV_FOLDS
            
        print(f"Iniciando validação cruzada (k={k})...")
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{k}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Criar novo modelo para cada fold
            modelo_fold = self.criar_modelo(X.shape[1])
            
            # Treinar com menos épocas para CV
            modelo_fold.fit(
                X_train_fold, y_train_fold,
                epochs=self.config.CV_EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0
            )
            
            # Avaliar
            score = modelo_fold.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
            scores.append(score)
            print(f"Acurácia Fold {fold + 1}: {score:.4f}")
        
        print(f"✓ Validação cruzada concluída!")
        print(f"Acurácia média: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        return scores
    
    def calcular_importancia_features(self, X_test, y_test):
        """Calcula a importância das features usando permutation importance manual."""
        print("Calculando importância das features...")
        
        # Acurácia base (sem permutação)
        y_pred_base = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
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
                y_pred_perm = np.argmax(self.model.predict(X_perm, verbose=0), axis=1)
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
    
    def extrair_embeddings(self, X):
        """Extrai embeddings da penúltima camada."""
        print("Extraindo embeddings...")
        
        # Primeiro, fazer uma predição dummy para construir o modelo
        _ = self.model.predict(X[:1], verbose=0)
        
        # Agora podemos criar o extrator de features da penúltima camada
        try:
            # Tentar acessar a penúltima camada (antes da softmax)
            extrator_features = keras.Model(inputs=self.model.input,
                                           outputs=self.model.layers[-2].output)
        except:
            # Se não conseguir, usar uma abordagem alternativa
            print("Usando abordagem alternativa para extrair embeddings...")
            # Criar um modelo temporário sem a última camada
            temp_model = keras.Sequential()
            for layer in self.model.layers[:-1]:  # Todas exceto a última
                temp_model.add(layer)
            
            # Fazer uma predição para construir o modelo temporário
            _ = temp_model.predict(X[:1], verbose=0)
            extrator_features = temp_model
        
        # Extrair embeddings
        embeddings = extrator_features.predict(X, verbose=0)
        
        return embeddings
    
    def teste_robustez(self, X_test, y_test, niveis_ruido=None):
        """Testa robustez adicionando ruído gaussiano."""
        if niveis_ruido is None:
            niveis_ruido = self.config.NOISE_LEVELS
            
        print("Testando robustez com ruído gaussiano...")
        
        resultados = []
        
        # Teste sem ruído
        y_pred_sem_ruido = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        acc_sem_ruido = accuracy_score(y_test, y_pred_sem_ruido)
        resultados.append(('Sem ruído', 0.0, acc_sem_ruido))
        print(f"Acurácia sem ruído: {acc_sem_ruido:.4f}")
        
        # Testes com ruído
        for nivel in niveis_ruido:
            ruido = np.random.normal(0, nivel, X_test.shape)
            X_test_ruido = X_test + ruido
            
            y_pred_ruido = np.argmax(self.model.predict(X_test_ruido, verbose=0), axis=1)
            acc_ruido = accuracy_score(y_test, y_pred_ruido)
            resultados.append((f'Ruído σ={nivel}', nivel, acc_ruido))
            print(f"Acurácia com ruído σ={nivel}: {acc_ruido:.4f}")
        
        return resultados
    
    def executar_treinamento(self, data_path, feature_names=None):
        """Executa apenas o treinamento e avaliação básica."""
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
        
        # 5. Criar e treinar modelo
        print("\n4. CRIAÇÃO E TREINAMENTO DO MODELO")
        print("-"*40)
        tempo_treinamento_inicio = time.time()
        
        self.model = self.criar_modelo(X.shape[1])
        self.treinar_modelo(X_train_split, y_train_split, X_val, y_val)
        
        tempo_treinamento = time.time() - tempo_treinamento_inicio
        print(f"⏱️  Tempo de treinamento: {tempo_treinamento:.2f}s")
        
        # 6. Validação cruzada
        print("\n5. VALIDAÇÃO CRUZADA")
        print("-"*40)
        cv_scores = self.validacao_cruzada(X_train, y_train)
        
        # 7. Avaliação final
        print("\n6. AVALIAÇÃO FINAL")
        print("-"*40)
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
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
            'cv_scores': cv_scores,
            'tempo_total': tempo_total
        }