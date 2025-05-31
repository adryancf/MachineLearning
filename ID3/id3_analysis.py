import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Importar a implementação do ID3
from id3 import ID3Classifier, preprocess_data

def analisar_matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true))
    
    print("=== ANÁLISE DETALHADA DA MATRIZ ===")
    print("Matriz de Confusão:")
    print(cm)
    
    for i, classe in enumerate(classes):
        tp = cm[i][i]  # Verdadeiros positivos
        fp = sum(cm[:, i]) - tp  # Falsos positivos
        fn = sum(cm[i, :]) - tp  # Falsos negativos
        
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nClasse {classe}:")
        print(f"  Precisão: {precisao:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  Erros mais comuns:")
        
        # Mostra onde esta classe é mais confundida
        erros = [(j, cm[i][j]) for j in range(len(classes)) if i != j and cm[i][j] > 0]
        for j, count in sorted(erros, key=lambda x: x[1], reverse=True):
            print(f"    {count}x confundida com {classes[j]}")

def criar_imagem_matriz_detalhada(y_test, y_pred, filename='matriz_confusao_detalhada.png'):
    """Cria matriz de confusão com métricas detalhadas"""
    
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(set(y_test))
    
    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matriz de confusão principal
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=0.5, ax=ax1)
    
    ax1.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predição', fontweight='bold')
    ax1.set_ylabel('Valor Real', fontweight='bold')
    
    # Tabela de métricas por classe
    metricas_dados = []
    for i, classe in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
        
        metricas_dados.append([classe, f"{precisao:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
    
    # Criar tabela
    ax2.axis('tight')
    ax2.axis('off')
    
    tabela = ax2.table(cellText=metricas_dados,
                      colLabels=['Classe', 'Precisão', 'Recall', 'F1-Score'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0.3, 1, 0.4])
    
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1, 1.5)
    
    # Aplicar formatação apenas nas células que existem
    for pos in tabela._cells.keys():
        row, col = pos
        if row == 0:  # Linha do cabeçalho
            tabela[pos].set_facecolor('#4CAF50')
            tabela[pos].set_text_props(weight='bold', color='white')
        elif col == 0:  # Coluna do cabeçalho
            tabela[pos].set_facecolor('#2196F3')
            tabela[pos].set_text_props(weight='bold', color='white')
    
    # Métricas gerais com indicadores de qualidade
    acuracia = accuracy_score(y_test, y_pred)
    f1_geral = f1_score(y_test, y_pred, average='weighted')
    
    # Função para avaliar qualidade das métricas
    def avaliar_metrica(valor, tipo='acuracia'):
        if tipo == 'acuracia':
            if valor >= 0.90: return "[EXCELENTE]", '#2E7D32'
            elif valor >= 0.80: return "[BOM]", '#F57F17'
            elif valor >= 0.70: return "[REGULAR]", '#E65100'
            else: return "[RUIM]", '#C62828'
        else:  # f1_score
            if valor >= 0.85: return "[EXCELENTE]", '#2E7D32'
            elif valor >= 0.75: return "[BOM]", '#F57F17'
            elif valor >= 0.65: return "[REGULAR]", '#E65100'
            else: return "[RUIM]", '#C62828'
    
    status_acuracia, cor_acuracia = avaliar_metrica(acuracia, 'acuracia')
    status_f1, cor_f1 = avaliar_metrica(f1_geral, 'f1_score')
    
    # Texto com indicadores de qualidade
    texto_metricas = (
        f'Acuracia Geral: {acuracia:.4f} - {status_acuracia}\n'
        f'F1-Score Geral: {f1_geral:.4f} - {status_f1}\n\n'
        f'Criterios de Avaliacao:\n'
        f'Acuracia: [EXCELENTE]>=90% [BOM]80-89% [REGULAR]70-79% [RUIM]<70%\n'
        f'F1-Score: [EXCELENTE]>=85% [BOM]75-84% [REGULAR]65-74% [RUIM]<65%'
    )
    
    ax2.text(0.5, 0.05, texto_metricas,
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='lightblue', alpha=0.9))
    
    plt.suptitle('Analise Completa da Matriz de Confusao - ID3', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Análise completa salva como '{filename}'")
    print(f"📈 Avaliação do modelo:")
    print(f"   Acurácia: {acuracia:.4f} - {status_acuracia}")
    print(f"   F1-Score: {f1_geral:.4f} - {status_f1}")

def analisar_importancia_atributos(model, X_train, filename='importancia_atributos.png'):
    """Analisa e visualiza a importância dos atributos na árvore ID3"""
    
    def calcular_importancia(node, depth=0, importancia_dict=None):
        if importancia_dict is None:
            importancia_dict = {}
        
        if node.value is not None:  # Nó folha
            return importancia_dict
        
        # Peso baseado na profundidade (nós mais próximos da raiz são mais importantes)
        peso = 1.0 / (depth + 1)
        
        if node.feature in importancia_dict:
            importancia_dict[node.feature] += peso
        else:
            importancia_dict[node.feature] = peso
        
        # Recursão para filhos
        calcular_importancia(node.left, depth + 1, importancia_dict)
        calcular_importancia(node.right, depth + 1, importancia_dict)
        
        return importancia_dict
    
    # Calcular importâncias
    importancias = calcular_importancia(model.tree)
    
    # Normalizar para percentuais
    total = sum(importancias.values())
    importancias_norm = {attr: (valor/total)*100 for attr, valor in importancias.items()}
    
    # Ordenar por importância
    attrs_ordenados = sorted(importancias_norm.items(), key=lambda x: x[1], reverse=True)
    
    # Preparar dados para visualização
    atributos = [item[0] for item in attrs_ordenados]
    valores = [item[1] for item in attrs_ordenados]
    
    # Criar visualização com melhor layout
    fig = plt.figure(figsize=(18, 12))
    
    # Gráfico de barras horizontal (parte superior esquerda)
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    
    cores = plt.cm.viridis(np.linspace(0, 1, len(atributos)))
    barras = ax1.barh(atributos, valores, color=cores, height=0.6)
    
    ax1.set_xlabel('Importancia (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Atributos', fontweight='bold', fontsize=12)
    ax1.set_title('Importancia dos Atributos na Arvore ID3', fontweight='bold', fontsize=14, pad=20)
    ax1.grid(axis='x', alpha=0.3)
    
    # Ajustar limite do eixo X para dar espaço aos valores
    max_valor = max(valores)
    ax1.set_xlim(0, max_valor * 1.15)
    
    # Adicionar valores nas barras
    for i, (barra, valor) in enumerate(zip(barras, valores)):
        ax1.text(valor + max_valor * 0.01, i, f'{valor:.1f}%', 
                va='center', fontweight='bold', fontsize=10)
    
    # Gráfico de pizza (parte superior direita)
    ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(atributos)))
    wedges, texts, autotexts = ax2.pie(valores, labels=atributos, autopct='%1.1f%%',
                                      colors=colors_pie, startangle=90,
                                      textprops={'fontsize': 9, 'fontweight': 'bold'})
    
    ax2.set_title('Distribuicao da Importancia dos Atributos', fontweight='bold', fontsize=14, pad=20)
    
    # Melhorar legibilidade do gráfico de pizza
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    # Tabela de informações detalhadas (parte inferior, ocupando toda a largura)
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax3.axis('off')
    
    # Classificar atributos por importância
    if valores[0] >= 40:
        nivel_principal = "[CRITICO]"
        cor_nivel = '#C62828'
    elif valores[0] >= 25:
        nivel_principal = "[ALTO]" 
        cor_nivel = '#F57F17'
    else:
        nivel_principal = "[MODERADO]"
        cor_nivel = '#2E7D32'
    
    info_text = f"""ANALISE DETALHADA:

Atributo mais importante: {atributos[0]} ({valores[0]:.1f}%)
Atributo menos importante: {atributos[-1]} ({valores[-1]:.1f}%)
Diferenca: {valores[0] - valores[-1]:.1f}%

Nivel de concentracao: {nivel_principal}

INTERPRETACAO:"""
    
    if valores[0] >= 40:
        info_text += """
• Modelo muito dependente de poucos atributos
• Risco de overfitting
• Considere balanceamento dos dados"""
    elif valores[0] >= 25:
        info_text += """
• Boa hierarquia de importancia
• Atributos bem distribuidos  
• Modelo equilibrado"""
    else:
        info_text += """
• Importancia muito distribuida
• Todos atributos contribuem igualmente
• Pode indicar complexidade desnecessaria"""
    
    # Adicionar caixa de texto informativa
    props = dict(boxstyle='round,pad=1.0', facecolor='lightgray', alpha=0.8, edgecolor='gray')
    ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes, fontsize=11,
             ha='center', va='center', bbox=props, fontweight='normal')
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.08, right=0.95, hspace=0.3, wspace=0.3)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Retornar dados para uso posterior
    return {
        'importancias_percentual': importancias_norm,
        'ranking': attrs_ordenados,
        'atributo_principal': atributos[0],
        'importancia_principal': valores[0]
    }

def visualizar_embeddings_pca(X_test, y_test, model, filename='id3_embeddings_pca.png'):
    """Cria visualização PCA dos dados de teste com predições do ID3"""
    
    print("\n📊 GERANDO VISUALIZAÇÃO PCA...")
    tempo_inicio_pca = time.time()
    
    # Padronizar dados para PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Aplicar PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Obter predições
    y_pred = model.predict(X_test)
    
    # Criar visualização
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Classes reais
    classes = sorted(set(y_test))
    cores = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, classe in enumerate(classes):
        mask = np.array(y_test) == classe
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[cores[i]], label=f'Classe {classe}', 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)', fontweight='bold')
    ax1.set_title('Análise PCA - Classes Reais', fontweight='bold', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Adicionar informações sobre variância explicada
    variancia_total = sum(pca.explained_variance_ratio_)
    tempo_pca = time.time() - tempo_inicio_pca
    
    plt.suptitle(f'Análise PCA - ID3 (Variância: {variancia_total:.1%}, Tempo: {tempo_pca:.2f}s)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Visualização PCA salva como '{filename}'")
    print(f"📈 Variância explicada total: {variancia_total:.1%}")
    print(f"⏱️  Tempo de processamento PCA: {tempo_pca:.2f}s")
    
    return {
        'pca_components': X_pca,
        'explained_variance': pca.explained_variance_ratio_,
        'total_variance': variancia_total,
        'processing_time': tempo_pca
    }

def visualizar_embeddings_tsne(X_test, y_test, model, filename='id3_embeddings_tsne.png'):
    """Cria visualização t-SNE dos dados de teste com predições do ID3"""
    
    print("\n🔍 GERANDO VISUALIZAÇÃO T-SNE...")
    print("⚠️  t-SNE pode demorar alguns minutos para datasets maiores...")
    
    inicio_tsne = time.time()
    
    # Padronizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Aplicar t-SNE com parâmetros otimizados
    perplexity = min(30, len(X_test) // 4, 50)  # Perplexity dinâmica
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                n_iter=1000, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X_scaled)
    
    tempo_tsne = time.time() - inicio_tsne
    
    # Obter predições
    y_pred = model.predict(X_test)
    
    # Criar visualização
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Classes reais
    classes = sorted(set(y_test))
    cores = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, classe in enumerate(classes):
        mask = np.array(y_test) == classe
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[cores[i]], label=f'Classe {classe}', 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('t-SNE Dimensão 1', fontweight='bold')
    ax1.set_ylabel('t-SNE Dimensão 2', fontweight='bold')
    ax1.set_title('t-SNE - Classes Reais', fontweight='bold', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.suptitle(f't-SNE - ID3 (Perplexity: {perplexity}, Tempo: {tempo_tsne:.1f}s)', 
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Visualização t-SNE salva como '{filename}'")
    print(f"⏱️  Tempo de processamento t-SNE: {tempo_tsne:.2f}s")
    print(f"🔧 Perplexity utilizada: {perplexity}")
    
    return {
        'tsne_components': X_tsne,
        'processing_time': tempo_tsne,
        'perplexity_used': perplexity
    }

def teste_robustez_id3(model, X_test, y_test, niveis_ruido=[0.01, 0.05, 0.1, 0.2], filename='id3_robustez.png'):
    """Testa robustez do modelo ID3 com diferentes níveis de ruído"""
    
    print("\n🛡️  TESTE DE ROBUSTEZ...")
    print("📊 Testando robustez com ruído gaussiano em dados discretizados...")
    
    tempo_inicio_robustez = time.time()
    
    # Obter predição sem ruído
    y_pred_original = model.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    
    resultados = []
    accuracies = []
    
    # Adicionar resultado sem ruído
    resultados.append({
        'nivel_ruido': 0.0,
        'accuracy_media': accuracy_original,
        'accuracy_std': 0.0,
        'perda_performance': 0.0
    })
    accuracies.append(accuracy_original)
    
    print(f"   📈 Accuracy original (sem ruído): {accuracy_original:.4f}")
    
    for nivel in niveis_ruido:
        print(f"   🔍 Testando ruído {nivel*100:.0f}%...")
        
        scores_nivel = []
        
        # Múltiplas execuções para cada nível
        for execucao in range(10):
            try:
                # Para dados discretizados (quartis), aplicar ruído de forma inteligente
                X_test_ruido = X_test.copy()
                
                for col in X_test_ruido.columns:
                    # Obter valores únicos para essa coluna (quartis: 0, 1, 2, 3)
                    valores_unicos = sorted(X_test[col].unique())
                    n_quartis = len(valores_unicos)
                    
                    # Gerar ruído proporcional ao número de quartis
                    ruido_col = np.random.normal(0, nivel * n_quartis, len(X_test_ruido))
                    
                    # Aplicar ruído e manter dentro dos limites válidos
                    valores_com_ruido = X_test_ruido[col].values + ruido_col
                    
                    # Re-discretizar: arredondar e limitar aos quartis válidos
                    valores_discretizados = np.round(valores_com_ruido).astype(int)
                    valores_discretizados = np.clip(valores_discretizados, 
                                                   min(valores_unicos), 
                                                   max(valores_unicos))
                    
                    X_test_ruido[col] = valores_discretizados
                
                # Predição com ruído
                y_pred_ruido = model.predict(X_test_ruido)
                acc_ruido = accuracy_score(y_test, y_pred_ruido)
                scores_nivel.append(acc_ruido)
                
            except Exception as e:
                print(f"     ⚠️  Erro na execução {execucao+1}: {str(e)}")
                scores_nivel.append(0.0)  # Em caso de erro, accuracy = 0
        
        # Calcular estatísticas para este nível
        acc_media = np.mean(scores_nivel)
        acc_std = np.std(scores_nivel)
        perda_performance = (accuracy_original - acc_media) / accuracy_original * 100
        
        accuracies.append(acc_media)
        resultados.append({
            'nivel_ruido': nivel,
            'accuracy_media': acc_media,
            'accuracy_std': acc_std,
            'perda_performance': perda_performance
        })
        
        print(f"     📊 Accuracy: {acc_media:.3f} ± {acc_std:.3f} (Perda: {perda_performance:.1f}%)")
    
    tempo_robustez = time.time() - tempo_inicio_robustez
    
    # Visualização
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Gráfico de accuracy vs ruído
    niveis_percent = [r['nivel_ruido']*100 for r in resultados]
    accuracies_mean = [r['accuracy_media'] for r in resultados]
    accuracies_std = [r['accuracy_std'] for r in resultados]
    
    ax1.errorbar(niveis_percent, accuracies_mean, yerr=accuracies_std, 
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax1.axhline(y=accuracy_original, color='red', linestyle='--', 
               label=f'Sem ruído: {accuracy_original:.3f}', linewidth=2)
    
    ax1.set_xlabel('Nível de Ruído (%)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Robustez do ID3 vs Nível de Ruído', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Adicionar anotações com valores
    for i, (nivel, acc, std) in enumerate(zip(niveis_percent, accuracies_mean, accuracies_std)):
        ax1.annotate(f'{acc:.3f}', (nivel, acc), 
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Adicionar informações no gráfico
    perda_maxima = max([r['perda_performance'] for r in resultados[1:]])
    if perda_maxima < 15:
        status_robustez = "🛡️ MODELO ROBUSTO"
        cor_status = '#2E7D32'
    elif perda_maxima < 30:
        status_robustez = "⚖️ MODELO MODERADO"
        cor_status = '#F57F17'
    else:
        status_robustez = "⚠️ MODELO FRÁGIL"
        cor_status = '#C62828'
    
    # Caixa de informações
    info_text = f"""ANÁLISE DE ROBUSTEZ:

• Accuracy baseline: {accuracy_original:.3f}
• Perda máxima: {perda_maxima:.1f}%
• Amostras testadas: {len(X_test)}
• Status: {status_robustez}

Critérios:
• Robusto: < 15% perda
• Moderado: 15-30% perda  
• Frágil: > 30% perda"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='lightblue', 
                      alpha=0.9, edgecolor=cor_status, linewidth=2))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Análise da robustez
    print(f"\n📊 ANÁLISE DE ROBUSTEZ:")
    print(f"{'─'*60}")
    print(f"🎯 Accuracy baseline: {accuracy_original:.4f}")
    
    for resultado in resultados[1:]:  # Excluir sem ruído
        nivel = resultado['nivel_ruido']
        acc = resultado['accuracy_media']
        perda = resultado['perda_performance']
        
        if perda < 10:
            status = "🟢 EXCELENTE"
        elif perda < 20:
            status = "🟡 BOM"
        elif perda < 30:
            status = "🟠 MODERADO"
        else:
            status = "🔴 FRÁGIL"
        
        print(f"📈 Ruído {nivel*100:2.0f}%: {acc:.3f} (Perda: {perda:5.1f}%) {status}")
    
    # Classificação geral de robustez
    perda_media = np.mean([r['perda_performance'] for r in resultados[1:]])
    if perda_media < 15:
        classificacao = "MODELO ROBUSTO PARA PRODUÇÃO"
        emoji = "🛡️"
    elif perda_media < 30:
        classificacao = "MODELO MODERADAMENTE ROBUSTO"
        emoji = "⚖️"
    else:
        classificacao = "MODELO FRÁGIL - CUIDADO EM PRODUÇÃO"
        emoji = "⚠️"
    
    print(f"{'─'*60}")
    print(f"{emoji} CLASSIFICAÇÃO: {classificacao}")
    print(f"📊 Perda média de performance: {perda_media:.1f}%")
    print(f"📊 Perda máxima observada: {perda_maxima:.1f}%")
    print(f"✅ Análise de robustez salva como '{filename}'")
    
    return {
        'accuracy_baseline': accuracy_original,
        'resultados_ruido': resultados,
        'classificacao_robustez': classificacao,
        'perda_media': perda_media,
        'perda_maxima': perda_maxima,
        'amostras_testadas': len(X_test)  # CORRIGIDO: era len(X_val)
    }

def preprocess_data_separate_files(labeled_file, unlabeled_file):
    """Processa arquivos separados: um com labels e outro sem labels"""
    
    # Carregar arquivo com labels
    df_labeled = pd.read_csv(labeled_file, header=None)
    
    # Definir nomes das colunas
    feature_cols = [f'feature_{i}' for i in range(1, 7)]
    df_labeled.columns = ['id'] + feature_cols + ['target']
    
    # Remover coluna ID e separar features e target
    X_train = df_labeled[feature_cols]
    y_train = df_labeled['target']
    
    # Carregar arquivo sem labels
    df_unlabeled = pd.read_csv(unlabeled_file, header=None)
    df_unlabeled.columns = ['id'] + feature_cols
    
    # Separar apenas features (sem ID)
    X_test = df_unlabeled[feature_cols]
    
    # Aplicar mesmo pré-processamento em ambos
    from scipy import stats
    
    # Detectar e remover outliers do conjunto de treino
    z_scores = np.abs(stats.zscore(X_train))
    mask_outliers = (z_scores < 3).all(axis=1)
    X_train = X_train[mask_outliers]
    y_train = y_train[mask_outliers]
    
    # Discretização baseada nos quartis do conjunto de treino
    for col in feature_cols:
        # Calcular quartis baseado apenas nos dados de treino
        quartis = pd.qcut(X_train[col], q=4, labels=False, duplicates='drop')
        X_train[col] = quartis
        
        # Aplicar mesma discretização nos dados de teste
        bins = pd.qcut(X_train[col], q=4, retbins=True, duplicates='drop')[1]
        X_test[col] = pd.cut(X_test[col], bins=bins, labels=False, include_lowest=True)
        
        # Tratar valores NaN que podem surgir da discretização
        X_test[col] = X_test[col].fillna(0)
    
    print(f"📊 Dados processados:")
    print(f"   • Treino (com labels): {len(X_train)} amostras")
    print(f"   • Teste (sem labels): {len(X_test)} amostras")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Classes no treino: {sorted(set(y_train))}")
    
    return X_train, X_test, y_train, None  # Retorna None para y_test (não existe)

def avaliar_modelo_sem_ground_truth(model, X_test, y_pred):
    """Avalia modelo quando não temos ground truth"""
    
    print("📊 AVALIAÇÃO SEM GROUND TRUTH...")
    
    # Análise da distribuição das predições
    distribuicao = Counter(y_pred)
    total_predicoes = len(y_pred)
    
    print(f"   📈 Distribuição das predições:")
    for classe, count in sorted(distribuicao.items()):
        percentual = (count / total_predicoes) * 100
        print(f"      Classe {classe}: {count} ({percentual:.1f}%)")
    
    # Análise de confiança baseada na consistência
    # (Para ID3, vamos usar a profundidade das predições como proxy de confiança)
    def calcular_confianca_predicao(model, X_test):
        """Calcula confiança baseada na profundidade da decisão"""
        confiancas = []
        
        def profundidade_decisao(sample, node, prof=0):
            if node.value is not None:
                return prof
            
            if sample[node.feature] <= node.threshold:
                return profundidade_decisao(sample, node.left, prof + 1)
            else:
                return profundidade_decisao(sample, node.right, prof + 1)
        
        for _, sample in X_test.iterrows():
            prof = profundidade_decisao(sample, model.tree)
            # Confiança inversamente proporcional à profundidade
            confianca = 1.0 / (prof + 1)
            confiancas.append(confianca)
        
        return confiancas
    
    confiancas = calcular_confianca_predicao(model, X_test)
    confianca_media = np.mean(confiancas)
    
    print(f"   🎯 Confiança média das predições: {confianca_media:.3f}")
    
    # Visualizar distribuição
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Distribuição das classes
    plt.subplot(1, 2, 1)
    classes = list(distribuicao.keys())
    counts = list(distribuicao.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Distribuição das Predições', fontweight='bold', fontsize=14)
    plt.xlabel('Classes Preditas', fontweight='bold')
    plt.ylabel('Número de Amostras', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Distribuição de confiança
    plt.subplot(1, 2, 2)
    plt.hist(confiancas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(confianca_media, color='red', linestyle='--', linewidth=2,
                label=f'Média: {confianca_media:.3f}')
    plt.title('Distribuição da Confiança', fontweight='bold', fontsize=14)
    plt.xlabel('Confiança', fontweight='bold')
    plt.ylabel('Frequência', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('id3_avaliacao_sem_gt_distribuicao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Análise salva como 'id3_avaliacao_sem_gt_distribuicao.png'")
    
    return {
        'distribuicao_classes': distribuicao,
        'confianca_media': confianca_media,
        'confiancas_individuais': confiancas,
        'total_predicoes': total_predicoes
    }

def visualizar_embeddings_producao(X_test, y_pred, metodo='pca'):
    """
    Visualiza embeddings dos dados de teste com as predições (sem ground truth)
    """
    if metodo == 'pca':
        print("📊 Gerando análise PCA dos dados de teste...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)
        
        pca = PCA(n_components=2, random_state=42)
        X_embedded = pca.fit_transform(X_scaled)
        
        filename = 'id3_teste_pca_producao.png'
        titulo = f'PCA - Dados de Teste com Predições (Var: {sum(pca.explained_variance_ratio_):.1%})'
        
    else:  # tsne
        print("🔍 Gerando análise t-SNE dos dados de teste...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)
        
        perplexity = min(30, len(X_test) // 4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_embedded = tsne.fit_transform(X_scaled)
        
        filename = 'id3_teste_tsne_producao.png'
        titulo = f't-SNE - Dados de Teste com Predições (Perplexity: {perplexity})'
    
    # Visualização
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = sorted(set(y_pred))
    cores = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, classe in enumerate(classes):
        mask = np.array(y_pred) == classe
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                  c=[cores[i]], label=f'Predição: Classe {classe}',
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_title(titulo, fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualização salva como '{filename}'")
    
    return X_embedded

def calcular_profundidade_arvore(node, profundidade=0):
    """
    Calcula a profundidade máxima da árvore
    """
    if node.value is not None:
        return profundidade
    
    prof_esquerda = calcular_profundidade_arvore(node.left, profundidade + 1)
    prof_direita = calcular_profundidade_arvore(node.right, profundidade + 1)
    
    return max(prof_esquerda, prof_direita)

def teste_robustez_producao(model, X_val, y_val, niveis_ruido=[0.01, 0.05, 0.1, 0.2], filename='id3_robustez_producao.png'):
    """
    Teste de robustez específico para cenário de produção usando dados de validação
    """
    print("🛡️  TESTE DE ROBUSTEZ - CENÁRIO DE PRODUÇÃO")
    print("📊 Testando robustez com ruído gaussiano em dados discretizados...")
    
    # Obter predição sem ruído
    y_pred_original = model.predict(X_val)
    accuracy_original = accuracy_score(y_val, y_pred_original)
    
    resultados = []
    accuracies = []
    
    # Adicionar resultado sem ruído
    resultados.append({
        'nivel_ruido': 0.0,
        'accuracy_media': accuracy_original,
        'accuracy_std': 0.0,
        'perda_performance': 0.0
    })
    accuracies.append(accuracy_original)
    
    print(f"   📈 Accuracy original (sem ruído): {accuracy_original:.4f}")
    
    for nivel in niveis_ruido:
        print(f"   🔍 Testando ruído {nivel*100:.0f}%...")
        
        scores_nivel = []
        
        # Múltiplas execuções para cada nível
        for execucao in range(5):  # Reduzido para ser mais rápido
            try:
                # Para dados discretizados (quartis), aplicar ruído de forma inteligente
                X_val_ruido = X_val.copy()
                
                for col in X_val_ruido.columns:
                    # Obter valores únicos para essa coluna (quartis: 0, 1, 2, 3)
                    valores_unicos = sorted(X_val[col].unique())
                    n_quartis = len(valores_unicos)
                    
                    # Gerar ruído proporcional ao número de quartis
                    ruido_col = np.random.normal(0, nivel * n_quartis, len(X_val_ruido))
                    
                    # Aplicar ruído e manter dentro dos limites válidos
                    valores_com_ruido = X_val_ruido[col].values + ruido_col
                    
                    # Re-discretizar: arredondar e limitar aos quartis válidos
                    valores_discretizados = np.round(valores_com_ruido).astype(int)
                    valores_discretizados = np.clip(valores_discretizados, 
                                                   min(valores_unicos), 
                                                   max(valores_unicos))
                    
                    X_val_ruido[col] = valores_discretizados
                
                # Predição com ruído
                y_pred_ruido = model.predict(X_val_ruido)
                acc_ruido = accuracy_score(y_val, y_pred_ruido)
                scores_nivel.append(acc_ruido)
                
            except Exception as e:
                print(f"     ⚠️  Erro na execução {execucao+1}: {str(e)}")
                scores_nivel.append(0.0)  # Em caso de erro, accuracy = 0
        
        # Calcular estatísticas para este nível
        acc_media = np.mean(scores_nivel)
        acc_std = np.std(scores_nivel)
        perda_performance = (accuracy_original - acc_media) / accuracy_original * 100
        
        accuracies.append(acc_media)
        resultados.append({
            'nivel_ruido': nivel,
            'accuracy_media': acc_media,
            'accuracy_std': acc_std,
            'perda_performance': perda_performance
        })
        
        print(f"     📊 Accuracy: {acc_media:.3f} ± {acc_std:.3f} (Perda: {perda_performance:.1f}%)")
    
    # Visualização
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Gráfico de accuracy vs ruído
    niveis_percent = [r['nivel_ruido']*100 for r in resultados]
    accuracies_mean = [r['accuracy_media'] for r in resultados]
    accuracies_std = [r['accuracy_std'] for r in resultados]
    
    ax1.errorbar(niveis_percent, accuracies_mean, yerr=accuracies_std, 
                marker='o', linewidth=3, markersize=10, capsize=8, capthick=3,
                color='#2E86AB', ecolor='#A23B72', alpha=0.8)
    ax1.axhline(y=accuracy_original, color='red', linestyle='--', 
               label=f'Baseline (sem ruído): {accuracy_original:.3f}', linewidth=2)
    
    ax1.set_xlabel('Nível de Ruído (%)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=14)
    ax1.set_title('Teste de Robustez - Cenário de Produção ID3', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=12, loc='best')
    ax1.set_ylim(0, min(1.1, max(accuracies_mean) * 1.1))
    
    # Adicionar anotações com valores
    for i, (nivel, acc, std) in enumerate(zip(niveis_percent, accuracies_mean, accuracies_std)):
        ax1.annotate(f'{acc:.3f}', (nivel, acc), 
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Adicionar informações no gráfico
    perda_maxima = max([r['perda_performance'] for r in resultados[1:]])
    if perda_maxima < 15:
        status_robustez = "🛡️ MODELO ROBUSTO"
        cor_status = '#2E7D32'
    elif perda_maxima < 30:
        status_robustez = "⚖️ MODELO MODERADO"
        cor_status = '#F57F17'
    else:
        status_robustez = "⚠️ MODELO FRÁGIL"
        cor_status = '#C62828'
    
    # Caixa de informações
    info_text = f"""ANÁLISE DE ROBUSTEZ - PRODUÇÃO:

• Accuracy baseline: {accuracy_original:.3f}
• Perda máxima: {perda_maxima:.1f}%
• Amostras testadas: {len(X_val)}
• Status: {status_robustez}

Critérios:
• Robusto: < 15% perda
• Moderado: 15-30% perda  
• Frágil: > 30% perda"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='lightblue', 
                      alpha=0.9, edgecolor=cor_status, linewidth=2))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Análise da robustez
    print(f"\n📊 ANÁLISE DE ROBUSTEZ - PRODUÇÃO:")
    print(f"{'─'*60}")
    print(f"🎯 Accuracy baseline: {accuracy_original:.4f}")
    
    for resultado in resultados[1:]:  # Excluir sem ruído
        nivel = resultado['nivel_ruido']
        acc = resultado['accuracy_media']
        perda = resultado['perda_performance']
        
        if perda < 10:
            status = "🟢 EXCELENTE"
        elif perda < 20:
            status = "🟡 BOM"
        elif perda < 30:
            status = "🟠 MODERADO"
        else:
            status = "🔴 FRÁGIL"
        
        print(f"📈 Ruído {nivel*100:2.0f}%: {acc:.3f} (Perda: {perda:5.1f}%) {status}")
    
    # Classificação geral de robustez
    perda_media = np.mean([r['perda_performance'] for r in resultados[1:]])
    if perda_media < 15:
        classificacao = "MODELO ROBUSTO PARA PRODUÇÃO"
        emoji = "🛡️"
    elif perda_media < 30:
        classificacao = "MODELO MODERADAMENTE ROBUSTO"
        emoji = "⚖️"
    else:
        classificacao = "MODELO FRÁGIL - CUIDADO EM PRODUÇÃO"
        emoji = "⚠️"
    
    print(f"{'─'*60}")
    print(f"{emoji} CLASSIFICAÇÃO: {classificacao}")
    print(f"📊 Perda média de performance: {perda_media:.1f}%")
    print(f"📊 Perda máxima observada: {perda_maxima:.1f}%")
    print(f"✅ Análise de robustez salva como '{filename}'")
    
    return {
        'accuracy_baseline': accuracy_original,
        'resultados_ruido': resultados,
        'classificacao_robustez': classificacao,
        'perda_media': perda_media,
        'perda_maxima': perda_maxima,
        'amostras_testadas': len(X_val)
    }

def executar_pipeline_completo_id3():
    """Executa pipeline completo do ID3 com medição de tempo (SEM JSON)"""
    
    # CRONÔMETRO PRINCIPAL
    tempo_inicio_total = time.time()
    
    print("="*80)
    print("🌳 PIPELINE COMPLETO - ALGORITMO ID3")
    print("="*80)
    
    # 1. PRÉ-PROCESSAMENTO
    print("\n1️⃣ PRÉ-PROCESSAMENTO DOS DADOS")
    print("─"*50)
    tempo_prep_inicio = time.time()
    
    try:
        X_train, X_test, y_train, y_test = preprocess_data("treino_sinais_vitais_com_label.txt")
        print(f"✅ Dados carregados com sucesso!")
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado. Tentando caminho alternativo...")
        try:
            X_train, X_test, y_train, y_test = preprocess_data("id3\\treino_sinais_vitais_com_label.txt")
            print(f"✅ Dados carregados com sucesso!")
        except:
            print(f"❌ Erro ao carregar dados. Verifique o caminho do arquivo.")
            return None
    
    tempo_preprocessamento = time.time() - tempo_prep_inicio
    
    print(f"📊 Informações do dataset:")
    print(f"   • Treino: {len(X_train)} amostras")
    print(f"   • Teste: {len(X_test)} amostras")
    print(f"   • Features: {len(X_train.columns)}")
    print(f"   • Classes: {len(set(y_train))}")
    print(f"⏱️  Tempo de pré-processamento: {tempo_preprocessamento:.2f}s")
    
    # 2. TREINAMENTO
    print("\n2️⃣ TREINAMENTO DO MODELO ID3")
    print("─"*50)
    tempo_treino_inicio = time.time()
    
    model = ID3Classifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    tempo_treinamento = time.time() - tempo_treino_inicio
    print(f"✅ Modelo treinado com sucesso!")
    print(f"⏱️  Tempo de treinamento: {tempo_treinamento:.2f}s")
    
    # 3. AVALIAÇÃO BÁSICA
    print("\n3️⃣ AVALIAÇÃO DO MODELO")
    print("─"*50)
    tempo_aval_inicio = time.time()
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    tempo_avaliacao_basica = time.time() - tempo_aval_inicio
    
    print(f"🎯 Acurácia: {accuracy:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"⏱️  Tempo de avaliação: {tempo_avaliacao_basica:.2f}s")
    
    # TEMPO ATÉ AVALIAÇÃO BÁSICA
    tempo_ate_avaliacao = time.time() - tempo_inicio_total
    
    print(f"\n⏱️  TEMPO TOTAL ATÉ AVALIAÇÃO: {tempo_ate_avaliacao:.2f}s ({tempo_ate_avaliacao/60:.1f} min)")
    
    # 4. EXPORTAÇÕES E VISUALIZAÇÕES
    print("\n4️⃣ GERAÇÃO DE RELATÓRIOS E VISUALIZAÇÕES")
    print("─"*50)
    tempo_viz_inicio = time.time()
    
    # Matriz de confusão detalhada
    print("📊 Gerando matriz de confusão...")
    criar_imagem_matriz_detalhada(y_test, y_pred, 'id3_matriz_confusao.png')
    
    # Visualização da árvore
    print("🌳 Gerando visualização da árvore...")
    model.plot_tree_graphviz('id3_arvore_decisao')
    
    # Análise de importância
    print("📈 Analisando importância dos atributos...")
    importancias = analisar_importancia_atributos(model, X_train, 'id3_importancia_atributos.png')
    
    tempo_relatorios = time.time() - tempo_viz_inicio
    
    # 5. VISUALIZAÇÕES AVANÇADAS
    print("\n5️⃣ VISUALIZAÇÕES AVANÇADAS")
    print("─"*50)
    tempo_viz_avancada_inicio = time.time()
    
    # PCA
    print("🔍 Gerando análise PCA...")
    pca_results = visualizar_embeddings_pca(X_test, y_test, model, 'id3_embeddings_pca.png')
    
    # t-SNE
    print("🎯 Gerando análise t-SNE...")
    tsne_results = visualizar_embeddings_tsne(X_test, y_test, model, 'id3_embeddings_tsne.png')
    
    tempo_viz_avancada = time.time() - tempo_viz_avancada_inicio
    
    # 6. TESTE DE ROBUSTEZ
    print("\n6️⃣ TESTE DE ROBUSTEZ")
    print("─"*50)
    tempo_robustez_inicio = time.time()
    
    robustez_results = teste_robustez_id3(model, X_test, y_test, filename='id3_robustez.png')
    
    tempo_robustez = time.time() - tempo_robustez_inicio
    
    # TEMPOS FINAIS
    tempo_total_completo = time.time() - tempo_inicio_total
    
    # RELATÓRIO FINAL
    print("\n" + "="*80)
    print("🎉 PIPELINE ID3 CONCLUÍDO COM SUCESSO!")
    print("="*80)
    
    print(f"\n📊 RESUMO DOS RESULTADOS:")
    print(f"{'─'*60}")
    print(f"🎯 Acurácia Final: {accuracy:.4f}")
    print(f"🎯 F1-Score Final: {f1:.4f}")
    print(f"🛡️  Robustez: {robustez_results['classificacao_robustez']}")
    print(f"📈 Perda média com ruído: {robustez_results['perda_media']:.1f}%")
    print(f"📊 Variância PCA explicada: {pca_results['total_variance']:.1%}")
    
    print(f"\n⏱️  RELATÓRIO DETALHADO DE TEMPOS:")
    print(f"{'─'*60}")
    print(f"🔧 Pré-processamento: {tempo_preprocessamento:.2f}s")
    print(f"🧠 Treinamento: {tempo_treinamento:.2f}s")
    print(f"📊 Avaliação básica: {tempo_avaliacao_basica:.2f}s")
    print(f"📄 Relatórios: {tempo_relatorios:.2f}s")
    print(f"🎨 Visualizações avançadas: {tempo_viz_avancada:.2f}s")
    print(f"🛡️  Teste de robustez: {tempo_robustez:.2f}s")
    print(f"{'─'*60}")
    print(f"⏱️  TEMPO ATÉ AVALIAÇÃO: {tempo_ate_avaliacao:.2f}s ({tempo_ate_avaliacao/60:.1f} min)")
    print(f"🏁 TEMPO TOTAL COMPLETO: {tempo_total_completo:.2f}s ({tempo_total_completo/60:.1f} min)")
    print(f"{'─'*60}")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    print(f"{'─'*40}")
    arquivos_gerados = [
        'id3_matriz_confusao.png',
        'id3_arvore_decisao.png', 
        'id3_importancia_atributos.png',
        'id3_embeddings_pca.png',
        'id3_embeddings_tsne.png',
        'id3_robustez.png'
    ]
    
    for arquivo in arquivos_gerados:
        if os.path.exists(arquivo):
            print(f"✅ {arquivo}")
        else:
            print(f"❌ {arquivo} (não gerado)")
    
    print("="*80)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'robustez': robustez_results,
        'pca_results': pca_results,
        'tsne_results': tsne_results,
        'importancias': importancias,
        'tempos': {
            'preprocessamento': tempo_preprocessamento,
            'treinamento': tempo_treinamento,
            'avaliacao_basica': tempo_avaliacao_basica,
            'relatorios': tempo_relatorios,
            'visualizacoes_avancadas': tempo_viz_avancada,
            'teste_robustez': tempo_robustez,
            'tempo_ate_avaliacao': tempo_ate_avaliacao,
            'tempo_total_completo': tempo_total_completo
        }
    }

def executar_pipeline_producao_id3():
    """
    Pipeline para cenário de produção: treino com dados rotulados, teste com dados não rotulados
    """
    # CRONÔMETRO PRINCIPAL
    tempo_inicio_total = time.time()
    
    print("="*80)
    print("🚀 PIPELINE ID3 - CENÁRIO DE PRODUÇÃO")
    print("🏷️  Treino: dados com labels | 🔍 Teste: dados sem labels")
    print("="*80)
    
    # 1. PRÉ-PROCESSAMENTO
    print("\n1️⃣ PRÉ-PROCESSAMENTO DOS DADOS")
    print("─"*50)
    tempo_prep_inicio = time.time()
    
    try:
        # Tentar carregar os arquivos
        labeled_file = "treino_sinais_vitais_com_label.txt"
        unlabeled_file = "treino_sinais_vitais_sem_label.txt"
        
        X_train, X_test, y_train, _ = preprocess_data_separate_files(labeled_file, unlabeled_file)
        print(f"✅ Dados carregados com sucesso!")
        
    except FileNotFoundError:
        print(f"❌ Arquivos não encontrados. Tentando caminho alternativo...")
        try:
            labeled_file = "id3/treino_sinais_vitais_com_label.txt"
            unlabeled_file = "id3/treino_sinais_vitais_sem_label.txt"
            X_train, X_test, y_train, _ = preprocess_data_separate_files(labeled_file, unlabeled_file)
            print(f"✅ Dados carregados com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return None
    
    tempo_preprocessamento = time.time() - tempo_prep_inicio
    print(f"⏱️  Tempo de pré-processamento: {tempo_preprocessamento:.2f}s")
    
    # 2. TREINAMENTO
    print("\n2️⃣ TREINAMENTO DO MODELO ID3")
    print("─"*50)
    tempo_treino_inicio = time.time()
    
    model = ID3Classifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    tempo_treinamento = time.time() - tempo_treino_inicio
    print(f"✅ Modelo treinado com sucesso!")
    print(f"⏱️  Tempo de treinamento: {tempo_treinamento:.2f}s")
    
    # 3. PREDIÇÃO NOS DADOS DE TESTE
    print("\n3️⃣ APLICAÇÃO DO MODELO NOS DADOS SEM LABELS")
    print("─"*50)
    tempo_pred_inicio = time.time()
    
    y_pred = model.predict(X_test)
    
    tempo_predicao = time.time() - tempo_pred_inicio
    print(f"✅ Predições realizadas!")
    print(f"⏱️  Tempo de predição: {tempo_predicao:.2f}s")
    print(f"📊 {len(y_pred)} amostras classificadas")
    
    # 4. AVALIAÇÃO SEM GROUND TRUTH
    print("\n4️⃣ ANÁLISE DAS PREDIÇÕES")
    print("─"*50)
    
    avaliacao_results = avaliar_modelo_sem_ground_truth(model, X_test, y_pred)
    
    # 5. TESTE DE ROBUSTEZ COM DADOS DE PRODUÇÃO
    print("\n5️⃣ TESTE DE ROBUSTEZ COM DADOS DE PRODUÇÃO")
    print("─"*50)
    tempo_robustez_inicio = time.time()
    
    # Usar dados de validação (parte dos dados de treino) para teste de robustez
    print("🔧 Criando conjunto de validação para teste de robustez...")
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_robustez, y_train_split, y_val_robustez = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Re-treinar modelo com dados reduzidos para teste mais realista
    print("🔄 Re-treinando modelo com 80% dos dados para teste de robustez...")
    model_robustez = ID3Classifier(max_depth=8, random_state=42)
    model_robustez.fit(X_train_split, y_train_split)
    
    # Executar teste de robustez
    robustez_results = teste_robustez_producao(
        model_robustez, X_val_robustez, y_val_robustez, 
        filename='id3_robustez_producao.png'
    )
    
    tempo_robustez = time.time() - tempo_robustez_inicio
    print(f"⏱️  Tempo de teste de robustez: {tempo_robustez:.2f}s")
    
    # 6. EXPORTAR PREDIÇÕES
    print("\n6️⃣ EXPORTAÇÃO DAS PREDIÇÕES")
    print("─"*50)
    
    # Criar DataFrame com resultados
    resultados_predicao = pd.DataFrame({
        'ID': range(1, len(y_pred) + 1),
        'Predicao': y_pred
    })
    
    # Salvar predições
    resultados_predicao.to_csv('predicoes_id3_producao.csv', index=False)
    print(f"✅ Predições salvas em 'predicoes_id3_producao.csv'")
    
    # 7. VISUALIZAÇÕES E RELATÓRIOS
    print("\n7️⃣ GERAÇÃO DE RELATÓRIOS")
    print("─"*50)
    tempo_viz_inicio = time.time()
    
    # Árvore de decisão
    print("🌳 Gerando visualização da árvore...")
    model.plot_tree_graphviz('id3_arvore_producao')
    
    # Análise de importância
    print("📈 Analisando importância dos atributos...")
    importancias = analisar_importancia_atributos(model, X_train, 'id3_importancia_producao.png')
    
    # Visualizações dos dados de teste
    print("🔍 Gerando análises dos dados de teste...")
    pca_results = visualizar_embeddings_producao(X_test, y_pred, 'pca')
    tsne_results = visualizar_embeddings_producao(X_test, y_pred, 'tsne')
    
    tempo_viz = time.time() - tempo_viz_inicio
    print(f"⏱️  Tempo de visualizações: {tempo_viz:.2f}s")
    
    tempo_total = time.time() - tempo_inicio_total
    
    # RELATÓRIO FINAL
    print("\n" + "="*80)
    print("🎉 PIPELINE DE PRODUÇÃO CONCLUÍDO!")
    print("="*80)
    
    print(f"\n📊 RESUMO:")
    print(f"{'─'*50}")
    print(f"🏷️  Amostras de treino: {len(X_train)}")
    print(f"🔍 Amostras classificadas: {len(y_pred)}")
    print(f"📈 Confiança média: {avaliacao_results['confianca_media']:.3f}")
    print(f"🌳 Profundidade da árvore: {calcular_profundidade_arvore(model.tree)}")
    print(f"🛡️  Robustez: {robustez_results['classificacao_robustez']}")
    print(f"📊 Perda média com ruído: {robustez_results['perda_media']:.1f}%")
    
    print(f"\n⏱️  TEMPOS:")
    print(f"{'─'*40}")
    print(f"🔧 Pré-processamento: {tempo_preprocessamento:.2f}s")
    print(f"🧠 Treinamento: {tempo_treinamento:.2f}s") 
    print(f"🎯 Predição: {tempo_predicao:.2f}s")
    print(f"🛡️  Teste robustez: {tempo_robustez:.2f}s")
    print(f"🎨 Visualizações: {tempo_viz:.2f}s")
    print(f"🏁 Tempo total: {tempo_total:.2f}s")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    print(f"{'─'*40}")
    arquivos_gerados = [
        'predicoes_id3_producao.csv',
        'id3_arvore_producao.png',
        'id3_importancia_producao.png',
        'id3_teste_pca_producao.png',
        'id3_teste_tsne_producao.png',
        'id3_avaliacao_sem_gt_distribuicao.png',
        'id3_robustez_producao.png'
    ]
    
    for arquivo in arquivos_gerados:
        if os.path.exists(arquivo):
            print(f"✅ {arquivo}")
        else:
            print(f"❌ {arquivo} (não gerado)")
    
    print("="*80)
    
    return {
        'model': model,
        'predicoes': y_pred,
        'avaliacao': avaliacao_results,
        'robustez': robustez_results,
        'importancias': importancias,
        'pca_results': pca_results,
        'tsne_results': tsne_results,
        'tempos': {
            'preprocessamento': tempo_preprocessamento,
            'treinamento': tempo_treinamento,
            'predicao': tempo_predicao,
            'robustez': tempo_robustez,
            'visualizacoes': tempo_viz,
            'total': tempo_total
        }
    }

# ADICIONAR if __name__ == "__main__" corrigido no final:
if __name__ == "__main__":
    print("Escolha o modo de execução:")
    print("1 - Pipeline tradicional (treino/teste do mesmo arquivo)")
    print("2 - Pipeline de produção (treino com labels, teste sem labels)")
    
    try:
        escolha = input("Digite sua escolha (1 ou 2): ").strip()
        
        if escolha == "1":
            print("🔄 Executando pipeline tradicional...")
            resultados = executar_pipeline_completo_id3()
        elif escolha == "2":
            print("🚀 Executando pipeline de produção...")
            resultados = executar_pipeline_producao_id3()
        else:
            print("❌ Escolha inválida. Executando pipeline de produção por padrão...")
            resultados = executar_pipeline_producao_id3()
            
        if resultados:
            print(f"\n🎯 Pipeline executado com sucesso!")
            if 'accuracy' in resultados:
                print(f"📊 Acurácia final: {resultados['accuracy']:.4f}")
            else:
                print(f"📊 {len(resultados['predicoes'])} predições realizadas")
            print(f"⏱️  Tempo total: {resultados['tempos']['total']:.2f}s")
        else:
            print(f"❌ Erro na execução do pipeline!")
            
    except KeyboardInterrupt:
        print("\n❌ Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"❌ Erro na execução: {str(e)}")
        print("🔄 Tentando pipeline de produção...")
        try:
            resultados = executar_pipeline_producao_id3()
            if resultados:
                print(f"✅ Pipeline de produção executado com sucesso!")
        except Exception as e2:
            print(f"❌ Erro no pipeline de produção: {str(e2)}")
