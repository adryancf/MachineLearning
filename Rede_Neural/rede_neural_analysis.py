import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rede_neural import RedeNeuralClassificador, ConfiguracaoRedeNeural
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

class RedeNeuralAnalysis:
    """Classe para análises e visualizações da Rede Neural."""
    
    def __init__(self, modelo_treinado):
        self.modelo = modelo_treinado
        self.config = ConfiguracaoRedeNeural()
        
        # Configurações de visualização
        self.FIGURE_SIZE_LARGE = (20, 12)
        self.FIGURE_SIZE_MEDIUM = (15, 5)
        self.FIGURE_SIZE_SMALL = (12, 9)
        self.PLOT_DPI = 100
    
    def plotar_curvas_aprendizado(self):
        """Plota as curvas de aprendizado."""
        # Verificar se o modelo foi treinado
        if not hasattr(self.modelo.model, 'training_history') or not self.modelo.model.training_history['loss']:
            print("Erro: Modelo não foi treinado ainda!")
            return
        
        history = self.modelo.model.training_history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_MEDIUM)
        
        # Curva de Loss
        ax1.plot(history['loss'], label='Treino', linewidth=2)
        ax1.plot(history['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Curva de Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curva de Acurácia
        ax2.plot(history['accuracy'], label='Treino', linewidth=2)
        ax2.plot(history['val_accuracy'], label='Validação', linewidth=2)
        ax2.set_title('Curva de Acurácia', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Acurácia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plotar_matriz_confusao(self, y_true, y_pred, classes=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']):
        """Plota a matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=self.FIGURE_SIZE_LARGE)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Quantidade de Predições'})
        plt.title('Matriz de Confusão - Rede Neural', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Classe Predita', fontsize=12, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=12, fontweight='bold')
        
        # Adicionar estatísticas
        accuracy = accuracy_score(y_true, y_pred)
        plt.figtext(0.02, 0.02, f'Acurácia Geral: {accuracy:.3f}', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plotar_importancia_features(self, importancia, feature_names=None):
        """Plota a importância das features com análise detalhada."""
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(importancia.importances_mean))]
        
        # Preparar dados
        importancias_mean = importancia.importances_mean
        importancias_std = importancia.importances_std
        
        # Ordenar por importância
        indices = np.argsort(importancias_mean)[::-1]
        
        # Criar figura com múltiplos subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Gráfico principal de barras
        ax1 = plt.subplot(2, 3, (1, 2))
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        bars = ax1.bar(range(len(indices)), 
                       importancias_mean[indices],
                       yerr=importancias_std[indices],
                       color=colors, alpha=0.8, capsize=5,
                       edgecolor='black', linewidth=0.5)
        
        ax1.set_title('Importância das Features - Rede Neural\n(Permutation Feature Importance)', 
                      fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Importância Média', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(indices)))
        ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + importancias_std[indices[i]],
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Análise estatística detalhada
        ax2 = plt.subplot(2, 3, 3)
        ax2.axis('off')
        
        # Calcular estatísticas
        max_imp = np.max(importancias_mean)
        min_imp = np.min(importancias_mean)
        mean_imp = np.mean(importancias_mean)
        std_imp = np.std(importancias_mean)
        
        # Feature mais importante
        most_important_idx = indices[0]
        least_important_idx = indices[-1]
        
        # Calcular variabilidade relativa
        cv_importances = importancias_std / np.maximum(importancias_mean, 1e-8)
        
        # Preparar valores para o texto de análise
        if len(indices) > 1:
            segundo_nome = feature_names[indices[1]]
            segundo_valor = importancias_mean[indices[1]]
            segundo_std = importancias_std[indices[1]]
        else:
            segundo_nome = 'N/A'
            segundo_valor = 0.0
            segundo_std = 0.0

        if len(indices) > 2:
            terceiro_nome = feature_names[indices[2]]
            terceiro_valor = importancias_mean[indices[2]]
            terceiro_std = importancias_std[indices[2]]
        else:
            terceiro_nome = 'N/A'
            terceiro_valor = 0.0
            terceiro_std = 0.0

        # Texto da análise
        analysis_text = f"""
ANÁLISE ESTATÍSTICA
{'='*25}

Resumo Geral:
• Importância Máxima: {max_imp:.4f}
• Importância Mínima: {min_imp:.4f}
• Importância Média: {mean_imp:.4f}
• Desvio Padrão: {std_imp:.4f}

Features Mais Relevantes:
• 1º: {feature_names[most_important_idx]}
  Valor: {importancias_mean[most_important_idx]:.4f} ± {importancias_std[most_important_idx]:.4f}
  
• 2º: {segundo_nome}
  Valor: {segundo_valor:.4f} ± {segundo_std:.4f}

• 3º: {terceiro_nome}
  Valor: {terceiro_valor:.4f} ± {terceiro_std:.4f}

Feature Menos Relevante:
• {feature_names[least_important_idx]}
  Valor: {importancias_mean[least_important_idx]:.4f} ± {importancias_std[least_important_idx]:.4f}

Variabilidade:
• CV Médio: {np.mean(cv_importances):.3f}
• Features Estáveis: {np.sum(cv_importances < 0.5)}
• Features Instáveis: {np.sum(cv_importances >= 0.5)}
"""

        ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 3. Ranking comparativo
        ax5 = plt.subplot(2, 3, 6)
        
        # Normalizar importâncias para porcentagem
        # Garantir que todas as importâncias sejam não-negativas para o pie chart
        importancias_positivas = np.maximum(importancias_mean, 0)
        
        # Se todas as importâncias são muito pequenas ou negativas, usar valores absolutos
        if np.sum(importancias_positivas) <= 1e-10:
            importancias_positivas = np.abs(importancias_mean)
        
        # Se ainda assim a soma for zero, usar valores uniformes
        if np.sum(importancias_positivas) <= 1e-10:
            importancias_positivas = np.ones_like(importancias_mean)
        
        total_importance_pos = np.sum(importancias_positivas)
        porcentagens = (importancias_positivas[indices] / total_importance_pos) * 100
        
        # Criar gráfico de pizza SEM os percentuais dentro
        wedges, texts = ax5.pie(porcentagens, labels=None, 
                               colors=colors, startangle=90)
        
        ax5.set_title('Contribuição Relativa (%)', fontweight='bold')
        
        # Criar legenda com percentuais
        labels_com_percentuais = [f'{feature_names[i]} ({porcentagens[j]:.1f}%)' 
                                 for j, i in enumerate(indices)]
        
        ax5.legend(wedges, labels_com_percentuais,
                   title="Features (% Contribuição)", 
                   loc="center left", 
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir análise textual detalhada
        self._imprimir_analise_importancia(importancias_mean, importancias_std, 
                                         feature_names, indices, cv_importances, mean_imp, std_imp)
    
    def _imprimir_analise_importancia(self, importancias_mean, importancias_std, 
                                    feature_names, indices, cv_importances, mean_imp, std_imp):
        """Imprime análise detalhada da importância das features."""
        print("\n" + "="*60)
        print("ANÁLISE DETALHADA DA IMPORTÂNCIA DAS FEATURES")
        print("="*60)
        
        print(f"\n📊 RANKING DE IMPORTÂNCIA:")
        print("-" * 40)
        
        # Para o ranking textual, usar as importâncias originais (podem ser negativas)
        total_importance = np.sum(np.abs(importancias_mean))
        
        for i, idx in enumerate(indices):
            porcentagem = (np.abs(importancias_mean[idx]) / total_importance) * 100
            stability = "Estável" if cv_importances[idx] < 0.5 else "Instável"
            
            # Indicar se a importância é negativa
            sinal = "+" if importancias_mean[idx] >= 0 else "-"
            
            print(f"{i+1:2d}º {feature_names[idx]:15s} | "
                  f"Valor: {sinal}{abs(importancias_mean[idx]):6.4f} ± {importancias_std[idx]:6.4f} | "
                  f"Contrib: {porcentagem:5.1f}% | {stability}")
        
        # Adicionar explicação sobre valores negativos se existirem
        valores_negativos = np.sum(importancias_mean < 0)
        if valores_negativos > 0:
            print(f"\n⚠️  ATENÇÃO: {valores_negativos} features têm importância negativa.")
            print("   Isso pode indicar que remover essas features melhora o modelo,")
            print("   ou que há multicolinearidade entre as features.")
        
        print(f"\n🔍 INSIGHTS:")
        print("-" * 40)
        
        # Identificar features dominantes
        threshold_high = mean_imp + std_imp
        dominant_features = [i for i in range(len(importancias_mean)) if importancias_mean[i] > threshold_high]
        
        if dominant_features:
            print(f"• Features dominantes (>{threshold_high:.3f}): {len(dominant_features)}")
            for idx in dominant_features:
                print(f"  - {feature_names[idx]}: {importancias_mean[idx]:.4f}")
        
        # Identificar features irrelevantes
        threshold_low = max(0, mean_imp - std_imp)
        irrelevant_features = [i for i in range(len(importancias_mean)) if importancias_mean[i] < threshold_low]
        
        if irrelevant_features:
            print(f"• Features menos relevantes (<{threshold_low:.3f}): {len(irrelevant_features)}")
            for idx in irrelevant_features:
                print(f"  - {feature_names[idx]}: {importancias_mean[idx]:.4f}")
        
        # Análise de estabilidade
        unstable_features = [i for i in range(len(cv_importances)) if cv_importances[i] >= 0.5]
        if unstable_features:
            print(f"• Features com alta variabilidade: {len(unstable_features)}")
            for idx in unstable_features:
                print(f"  - {feature_names[idx]}: CV = {cv_importances[idx]:.3f}")
        
        print(f"\n💡 RECOMENDAÇÕES:")
        print("-" * 40)
        
        if len(dominant_features) > 0:
            print(f"• Focar nas {len(dominant_features)} features mais importantes para otimização")
        
        if len(irrelevant_features) > 0:
            print(f"• Considerar remover {len(irrelevant_features)} features menos relevantes para simplificar o modelo")
        
        if len(unstable_features) > 0:
            print(f"• Investigar {len(unstable_features)} features com alta variabilidade - podem precisar de mais dados")
        
        # Calcular índice de concentração (Gini)
        importancias_norm = importancias_mean / np.sum(importancias_mean)
        importancias_sorted = np.sort(importancias_norm)
        n = len(importancias_sorted)
        gini = (2 * np.sum((np.arange(1, n+1) * importancias_sorted))) / (n * np.sum(importancias_sorted)) - (n+1)/n
        
        print(f"• Índice de concentração (Gini): {gini:.3f}")
        if gini > 0.5:
            print("  → Alta concentração: poucas features dominam o modelo")
        elif gini < 0.3:
            print("  → Baixa concentração: importância bem distribuída")
        else:
            print("  → Concentração moderada")
    
    def calcular_importancia_features(self, X_test, y_test):
        """Calcula a importância das features usando permutation importance manual."""
        print("Calculando importância das features...")
        
        # Acurácia base (sem permutação)
        y_pred_base = self.modelo.model.predict_classes(X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        # Importância por permutação
        importancias_mean = np.zeros(X_test.shape[1])
        importancias_std = np.zeros(X_test.shape[1])
        
        for i in range(X_test.shape[1]):
            # Permutar a i-ésima feature
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            
            # Calcular nova acurácia
            y_pred_perm = self.modelo.model.predict_classes(X_perm)
            acc_perm = accuracy_score(y_test, y_pred_perm)
            
            # Importância é a queda na acurácia
            importancias_mean[i] = acc_base - acc_perm
        
        # Para estabilidade, calcular desvio padrão em múltiplas permutações
        for i in range(X_test.shape[1]):
            quedas_acuracia = []
            
            for _ in range(30):  # 30 repetições
                X_perm = X_test.copy()
                np.random.shuffle(X_perm[:, i])
                
                y_pred_perm = self.modelo.model.predict_classes(X_perm)
                acc_perm = accuracy_score(y_test, y_pred_perm)
                
                quedas_acuracia.append(acc_base - acc_perm)
            
            importancias_std[i] = np.std(quedas_acuracia)
        
        return SimpleNamespace(importances_mean=importancias_mean, importances_std=importancias_std)
    
    def visualizar_embeddings(self, X, y, metodo='pca'):
        """Visualiza embeddings usando PCA ou t-SNE."""
        print(f"Gerando visualização de embeddings ({metodo.upper()})...")
        
        # Para rede neural manual, extrair da penúltima camada
        activations, _ = self.modelo.model.forward_propagation(X, training=False)
        embeddings = activations[-2]  # Penúltima camada (antes do softmax)
        
        # Se os embeddings têm mais de 2 dimensões, reduzir dimensionalidade
        if embeddings.shape[1] > 50:  # Se tem muitas features, usar PCA primeiro
            pca_pre = PCA(n_components=50, random_state=self.modelo.random_state)
            embeddings = pca_pre.fit_transform(embeddings)
        
        # Redução dimensional para visualização
        if metodo.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=self.modelo.random_state)
            titulo = 'Visualização de Embeddings - PCA'
        elif metodo.lower() == 'tsne':
            # Para t-SNE, ajustar perplexity baseado no tamanho da amostra
            perplexity = min(30, X.shape[0] // 4)
            reducer = TSNE(n_components=2, random_state=self.modelo.random_state, 
                          perplexity=perplexity, n_iter=1000)
            titulo = 'Visualização de Embeddings - t-SNE'
        else:
            raise ValueError("Método deve ser 'pca' ou 'tsne'")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plotar
        plt.figure(figsize=self.FIGURE_SIZE_MEDIUM)
        colors = ['red', 'blue', 'green', 'orange']
        classes = ['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']
        
        for i in range(4):
            mask = y == i
            if np.any(mask):  # Só plotar se existirem amostras desta classe
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=colors[i], label=classes[i], alpha=0.7, s=50)
        
        plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Componente 1', fontsize=12, fontweight='bold')
        plt.ylabel('Componente 2', fontsize=12, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plotar_teste_robustez(self, resultados_robustez):
        """Plota os resultados do teste de robustez."""
        plt.figure(figsize=self.FIGURE_SIZE_SMALL)
        niveis = [r[1] for r in resultados_robustez]
        acuracias = [r[2] for r in resultados_robustez]
        
        plt.plot(niveis, acuracias, 'bo-', linewidth=2, markersize=8)
        plt.title('Teste de Robustez - Ruído Gaussiano', fontsize=14, fontweight='bold')
        plt.xlabel('Desvio Padrão do Ruído', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        for i, (nome, nivel, acc) in enumerate(resultados_robustez):
            plt.annotate(f'{acc:.3f}', (nivel, acc), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def executar_analise_completa(self, data_path, feature_names=None):
        """Executa análise completa com visualizações."""
        import time
        
        print("="*60)
        print("ANÁLISE COMPLETA DA REDE NEURAL")
        print("="*60)
        
        tempo_inicio = time.time()
        
        # 1. Executar treinamento
        resultados = self.modelo.executar_treinamento(data_path, feature_names)
        
        # 2. Extrair dados dos resultados
        X_test = resultados['X_test']
        y_test = resultados['y_test']
        y_pred = resultados['y_pred']
        
        if feature_names is None:
            feature_names = ['Sinal Vital 1', 'Sinal Vital 2', 'Sinal Vital 3', 
                           'Sinal Vital 4', 'Sinal Vital 5', 'Sinal Vital 6']
        
        print("\n" + "="*60)
        print("INICIANDO ANÁLISES E VISUALIZAÇÕES")
        print("="*60)
        
        # 3. Visualizações
        print("\n1. CURVAS DE APRENDIZADO")
        print("-"*40)
        self.plotar_curvas_aprendizado()
        
        print("\n2. MATRIZ DE CONFUSÃO")
        print("-"*40)
        self.plotar_matriz_confusao(y_test, y_pred)
        
        print("\n3. IMPORTÂNCIA DAS FEATURES")
        print("-"*40)
        importancia = self.modelo.calcular_importancia_features(X_test, y_test)
        self.plotar_importancia_features(importancia, feature_names)
        
        print("\n4. VISUALIZAÇÃO DE EMBEDDINGS")
        print("-"*40)
        self.visualizar_embeddings(X_test, y_test, 'pca')
        self.visualizar_embeddings(X_test, y_test, 'tsne')
        
        print("\n5. TESTE DE ROBUSTEZ")
        print("-"*40)
        resultados_robustez = self.modelo.teste_robustez(X_test, y_test)
        self.plotar_teste_robustez(resultados_robustez)
        
        tempo_total = time.time() - tempo_inicio
        
        print("\n" + "="*60)
        print("ANÁLISE COMPLETA CONCLUÍDA!")
        print("="*60)
        print(f"Tempo total: {tempo_total:.2f}s ({tempo_total/60:.1f} min)")
        
        return {
            **resultados,
            'feature_importance': importancia,
            'robustez_results': resultados_robustez,
            'tempo_analise_completa': tempo_total
        }

# Exemplo de uso
if __name__ == "__main__":
    # Criar e treinar o modelo
    modelo = RedeNeuralClassificador()
    
    # Criar analisador
    analisador = RedeNeuralAnalysis(modelo)
    
    # Executar análise completa
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6']
    resultados = analisador.executar_analise_completa('treino_sinais_vitais_com_label.txt', feature_names)
    
    print("\nResultados finais:")
    print(f"Acurácia: {resultados['accuracy']:.4f}")
    print(f"F1-Score: {resultados['f1_score']:.4f}")
