import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

# Importar a implementação do Random Forest
from random_forest import RandomForestClassifier, preprocess_data

def analisar_matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true))
    
    print("=== ANÁLISE DETALHADA DA MATRIZ ===")
    print("Matriz de Confusão:")
    print(cm)
    
    for i, classe in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nClasse {classe}:")
        print(f"  Precisão: {precisao:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  Erros mais comuns:")
        
        erros = [(j, cm[i][j]) for j in range(len(classes)) if i != j and cm[i][j] > 0]
        for j, count in sorted(erros, key=lambda x: x[1], reverse=True):
            print(f"    {count}x confundida com {classes[j]}")

def exportar_resultados_json(y_true, y_pred, model, filename='resultados_random_forest.json'):
    """Exporta resultados para JSON (estruturado)"""
    
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true))
    
    resultados = {
        'metricas_gerais': {
            'acuracia': float(f"{accuracy_score(y_true, y_pred):.4f}"),
            'f1_score_weighted': float(f"{f1_score(y_true, y_pred, average='weighted'):.4f}"),
            'f1_score_macro': float(f"{f1_score(y_true, y_pred, average='macro'):.4f}")
        },
        'matriz_confusao': {
            'classes': classes,
            'matriz': cm.tolist()
        },
        'analise_por_classe': {},
        'parametros_random_forest': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_leaf': model.min_samples_leaf,
            'max_features': model.max_features,
            'bootstrap': model.bootstrap,
            'random_state': model.random_state
        },
        'estatisticas_floresta': {}
    }
    
    # Análise por classe
    for i, classe in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = sum(sum(cm)) - tp - fp - fn
        
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_individual = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
        
        erros = [(classes[j], int(cm[i][j])) for j in range(len(classes)) if i != j and cm[i][j] > 0]
        
        resultados['analise_por_classe'][classe] = {
            'verdadeiros_positivos': int(tp),
            'falsos_positivos': int(fp),
            'falsos_negativos': int(fn),
            'verdadeiros_negativos': int(tn),
            'precisao': float(f"{precisao:.4f}"),
            'recall': float(f"{recall:.4f}"),
            'f1_score': float(f"{f1_individual:.4f}"),
            'erros_comuns': erros
        }
    
    # Estatísticas da floresta
    def contar_nos_arvore(node):
        if node.value is not None:
            return 1, 1, 0
        total, folhas, internos = 1, 0, 1
        t_left, f_left, i_left = contar_nos_arvore(node.left)
        t_right, f_right, i_right = contar_nos_arvore(node.right)
        return total + t_left + t_right, folhas + f_left + f_right, internos + i_left + i_right
    
    def profundidade_arvore(node, prof=0):
        if node.value is not None:
            return prof
        return max(profundidade_arvore(node.left, prof + 1), 
                  profundidade_arvore(node.right, prof + 1))
    
    # Estatísticas de todas as árvores
    estatisticas_arvores = []
    for i, tree in enumerate(model.trees):
        total, folhas, internos = contar_nos_arvore(tree.tree)
        prof_max = profundidade_arvore(tree.tree)
        
        estatisticas_arvores.append({
            'arvore_id': i,
            'total_nos': total,
            'nos_folha': folhas,
            'nos_internos': internos,
            'profundidade_maxima': prof_max
        })
    
    # Resumo da floresta
    total_nos_media = np.mean([stat['total_nos'] for stat in estatisticas_arvores])
    prof_media = np.mean([stat['profundidade_maxima'] for stat in estatisticas_arvores])
    
    resultados['estatisticas_floresta'] = {
        'numero_arvores': len(model.trees),
        'nos_total_medio': float(f"{total_nos_media:.2f}"),
        'profundidade_media': float(f"{prof_media:.2f}"),
        'estatisticas_individuais': estatisticas_arvores[:5]  # Primeiras 5 árvores
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados exportados para '{filename}'")

def criar_imagem_matriz_detalhada(y_test, y_pred, filename='matriz_confusao_random_forest.png'):
    """Cria matriz de confusão com métricas detalhadas para Random Forest"""
    
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(set(y_test))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matriz de confusão principal
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=0.5, ax=ax1)
    
    ax1.set_title('Matriz de Confusão - Random Forest', fontsize=14, fontweight='bold')
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
    
    for pos in tabela._cells.keys():
        row, col = pos
        if row == 0:
            tabela[pos].set_facecolor('#4CAF50')
            tabela[pos].set_text_props(weight='bold', color='white')
        elif col == 0:
            tabela[pos].set_facecolor('#2196F3')
            tabela[pos].set_text_props(weight='bold', color='white')
    
    # Métricas gerais
    acuracia = accuracy_score(y_test, y_pred)
    f1_geral = f1_score(y_test, y_pred, average='weighted')
    
    def avaliar_metrica(valor, tipo='acuracia'):
        if tipo == 'acuracia':
            if valor >= 0.90: return "[EXCELENTE]", '#2E7D32'
            elif valor >= 0.80: return "[BOM]", '#F57F17'
            elif valor >= 0.70: return "[REGULAR]", '#E65100'
            else: return "[RUIM]", '#C62828'
        else:
            if valor >= 0.85: return "[EXCELENTE]", '#2E7D32'
            elif valor >= 0.75: return "[BOM]", '#F57F17'
            elif valor >= 0.65: return "[REGULAR]", '#E65100'
            else: return "[RUIM]", '#C62828'
    
    status_acuracia, cor_acuracia = avaliar_metrica(acuracia, 'acuracia')
    status_f1, cor_f1 = avaliar_metrica(f1_geral, 'f1_score')
    
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
    
    plt.suptitle('Analise Completa da Matriz de Confusao - Random Forest', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Análise completa salva como '{filename}'")
    print(f"📈 Avaliação do modelo:")
    print(f"   Acurácia: {acuracia:.4f} - {status_acuracia}")
    print(f"   F1-Score: {f1_geral:.4f} - {status_f1}")

def analisar_importancia_atributos(model, X_train, filename='importancia_atributos_rf.png'):
    """Analisa e visualiza a importância dos atributos na Random Forest"""
    
    # Usar as importâncias calculadas pelo modelo
    importancias_norm = {k: v*100 for k, v in model.feature_importances_.items()}
    
    # Ordenar por importância
    attrs_ordenados = sorted(importancias_norm.items(), key=lambda x: x[1], reverse=True)
    
    atributos = [item[0] for item in attrs_ordenados]
    valores = [item[1] for item in attrs_ordenados]
    
    # Criar visualização
    fig = plt.figure(figsize=(18, 12))
    
    # Gráfico de barras horizontal
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    
    cores = plt.cm.viridis(np.linspace(0, 1, len(atributos)))
    barras = ax1.barh(atributos, valores, color=cores, height=0.6)
    
    ax1.set_xlabel('Importancia (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Atributos', fontweight='bold', fontsize=12)
    ax1.set_title('Importancia dos Atributos na Random Forest', fontweight='bold', fontsize=14, pad=20)
    ax1.grid(axis='x', alpha=0.3)
    
    max_valor = max(valores) if valores else 1
    ax1.set_xlim(0, max_valor * 1.15)
    
    for i, (barra, valor) in enumerate(zip(barras, valores)):
        ax1.text(valor + max_valor * 0.01, i, f'{valor:.1f}%', 
                va='center', fontweight='bold', fontsize=10)
    
    # Gráfico de pizza
    ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(atributos)))
    wedges, texts, autotexts = ax2.pie(valores, labels=atributos, autopct='%1.1f%%',
                                      colors=colors_pie, startangle=90,
                                      textprops={'fontsize': 9, 'fontweight': 'bold'})
    
    ax2.set_title('Distribuicao da Importancia dos Atributos', fontweight='bold', fontsize=14, pad=20)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    # Informações detalhadas
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax3.axis('off')
    
    if valores:
        if valores[0] >= 40:
            nivel_principal = "[CRITICO]"
        elif valores[0] >= 25:
            nivel_principal = "[ALTO]" 
        else:
            nivel_principal = "[MODERADO]"
        
        info_text = f"""ANALISE DETALHADA DA RANDOM FOREST:

Numero de Arvores: {model.n_estimators}
Max Features por Arvore: {model.max_features}

Atributo mais importante: {atributos[0]} ({valores[0]:.1f}%)
Atributo menos importante: {atributos[-1]} ({valores[-1]:.1f}%)
Diferenca: {valores[0] - valores[-1]:.1f}%

Nivel de concentracao: {nivel_principal}

INTERPRETACAO:"""
        
        if valores[0] >= 40:
            info_text += """
• Modelo concentrado em poucos atributos
• Random Forest conseguiu identificar features dominantes
• Boa estabilidade das importancias"""
        elif valores[0] >= 25:
            info_text += """
• Excelente distribuicao de importancia
• Random Forest balanceou bem os atributos
• Modelo robusto e generalizado"""
        else:
            info_text += """
• Importancia muito distribuida
• Todos atributos contribuem similarmente
• Random Forest explorou bem toda informacao disponivel"""
    else:
        info_text = "Erro: Não foi possível calcular importâncias"
    
    props = dict(boxstyle='round,pad=1.0', facecolor='lightgray', alpha=0.8, edgecolor='gray')
    ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes, fontsize=11,
             ha='center', va='center', bbox=props, fontweight='normal')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.08, right=0.95, hspace=0.3, wspace=0.3)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return {
        'importancias_percentual': importancias_norm,
        'ranking': attrs_ordenados,
        'atributo_principal': atributos[0] if atributos else 'N/A',
        'importancia_principal': valores[0] if valores else 0
    }

def exportar_importancia_json(importancias_data, filename='importancia_atributos_rf.json'):
    """Exporta análise de importância para JSON"""
    
    resultado = {
        'analise_importancia_random_forest': {
            'metodologia': 'Agregação das importâncias de todas as árvores da floresta',
            'interpretacao': {
                'atributo_principal': importancias_data['atributo_principal'],
                'importancia_principal': round(importancias_data['importancia_principal'], 2),
                'distribuicao': 'equilibrada' if importancias_data['importancia_principal'] < 40 else 'concentrada'
            },
            'ranking_completo': [
                {
                    'posicao': i+1,
                    'atributo': attr,
                    'importancia_percentual': round(valor, 2)
                }
                for i, (attr, valor) in enumerate(importancias_data['ranking'])
            ],
            'metricas_resumo': {
                'total_atributos': len(importancias_data['ranking']),
                'concentracao_top3': round(sum([valor for _, valor in importancias_data['ranking'][:3]]), 2) if len(importancias_data['ranking']) >= 3 else 0,
                'distribuicao_uniforme': bool((max([valor for _, valor in importancias_data['ranking']]) - min([valor for _, valor in importancias_data['ranking']])) < 20) if importancias_data['ranking'] else False
            }
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Análise de importância exportada para '{filename}'")

def visualizar_embeddings(X, y, metodo='pca', filename_prefix='random_forest'):
    """Visualiza embeddings usando PCA ou t-SNE."""
    print(f"Gerando visualização de embeddings ({metodo.upper()})...")
    
    # Para Random Forest, usar as próprias features normalizadas
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(X)
    
    # Redução dimensional
    if metodo.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        titulo = 'Visualização do Espaço de Features - PCA (Random Forest)'
        filename = f'{filename_prefix}_embeddings_pca.png'
    elif metodo.lower() == 'tsne':
        perplexity = min(30, max(5, X.shape[0] // 4))
        reducer = TSNE(n_components=2, random_state=42, 
                      perplexity=perplexity, n_iter=1000)
        titulo = 'Visualização do Espaço de Features - t-SNE (Random Forest)'
        filename = f'{filename_prefix}_embeddings_tsne.png'
    else:
        raise ValueError("Método deve ser 'pca' ou 'tsne'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plotar
    plt.figure(figsize=(12, 9))
    colors = ['red', 'blue', 'green', 'orange']
    classes = ['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']
    
    # Converter y para numérico se for string
    y_numeric = pd.Series(y).astype(str).map({'1': 0, '2': 1, '3': 2, '4': 3})
    
    for i in range(4):
        mask = y_numeric == i
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[i], label=classes[i], alpha=0.7, s=50)
    
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Componente 1', fontsize=12, fontweight='bold')
    plt.ylabel('Componente 2', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Visualização salva como '{filename}'")

def teste_robustez(model, X_test, y_test, niveis_ruido=[0.1, 0.2, 0.3], filename='random_forest_robustez.png'):
    """Testa robustez adicionando ruído gaussiano."""
    print("Testando robustez com ruído gaussiano...")
    
    resultados = []
    
    # Teste sem ruído
    y_pred_sem_ruido = model.predict(X_test)
    acc_sem_ruido = accuracy_score(y_test, y_pred_sem_ruido)
    resultados.append(('Sem ruído', 0.0, acc_sem_ruido))
    print(f"Acurácia sem ruído: {acc_sem_ruido:.4f}")
    
    # Testes com ruído
    for nivel in niveis_ruido:
        # Gerar ruído gaussiano
        ruido = np.random.normal(0, nivel, X_test.shape)
        X_test_ruido = X_test + ruido
        
        # Predição com ruído
        y_pred_ruido = model.predict(X_test_ruido)
        acc_ruido = accuracy_score(y_test, y_pred_ruido)
        resultados.append((f'Ruído σ={nivel}', nivel, acc_ruido))
        print(f"Acurácia com ruído σ={nivel}: {acc_ruido:.4f}")
    
    # Plotar resultados
    plt.figure(figsize=(10, 6))
    niveis = [r[1] for r in resultados]
    acuracias = [r[2] for r in resultados]
    
    plt.plot(niveis, acuracias, 'go-', linewidth=2, markersize=8)
    plt.title('Teste de Robustez - Ruído Gaussiano (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Desvio Padrão do Ruído', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adicionar anotações com valores
    for i, (nome, nivel, acc) in enumerate(resultados):
        plt.annotate(f'{acc:.3f}', (nivel, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Adicionar linha de referência
    plt.axhline(y=acc_sem_ruido, color='red', linestyle='--', alpha=0.7, 
                label=f'Acurácia base: {acc_sem_ruido:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Teste de robustez salvo como '{filename}'")
    
    # Análise dos resultados
    degradacao_max = acc_sem_ruido - min(acuracias)
    print(f"\n📊 Análise de Robustez:")
    print(f"   • Degradação máxima: {degradacao_max:.4f}")
    
    if degradacao_max < 0.05:
        print("   • Modelo MUITO ROBUSTO ao ruído")
    elif degradacao_max < 0.10:
        print("   • Modelo ROBUSTO ao ruído")
    elif degradacao_max < 0.20:
        print("   • Modelo MODERADAMENTE ROBUSTO ao ruído")
    else:
        print("   • Modelo SENSÍVEL ao ruído")
    
    return resultados

def executar_pipeline_completo(model, X_train, X_test, y_train, y_test, nome_modelo='rf_balanceado'):
    """Executa o pipeline completo de análise com medição de tempo."""
    
    pipeline_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 EXECUTANDO PIPELINE COMPLETO - {nome_modelo.upper()}")
    print(f"{'='*60}")
    
    # 1. Treinamento COM CRONÔMETRO
    print("\n1. TREINAMENTO DO MODELO")
    print("-"*40)
    
    training_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - training_start
    
    print(f"\n📊 ESTATÍSTICAS DE TREINAMENTO:")
    print(f"   • Tempo total: {training_time:.2f}s")
    print(f"   • Árvores treinadas: {len(model.trees)}")
    print(f"   • Tempo médio/árvore: {training_time/len(model.trees):.3f}s")
    
    # 2. Avaliação básica COM CRONÔMETRO
    print("\n2. AVALIAÇÃO BÁSICA")
    print("-"*40)
    
    prediction_start = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - prediction_start
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"📈 Resultados:")
    print(f"   • Acurácia: {accuracy:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    print(f"   • Tempo de predição: {prediction_time:.3f}s")
    print(f"   • Predições/segundo: {len(X_test)/prediction_time:.1f}")
    
    # 3. Matriz de confusão
    print("\n3. MATRIZ DE CONFUSÃO")
    print("-"*40)
    matrix_start = time.time()
    criar_imagem_matriz_detalhada(y_test, y_pred, f'matriz_confusao_{nome_modelo}.png')
    matrix_time = time.time() - matrix_start
    print(f"   ⏱️  Tempo: {matrix_time:.2f}s")
    
    # 4. Análise de importância
    print("\n4. IMPORTÂNCIA DOS ATRIBUTOS")
    print("-"*40)
    importance_start = time.time()
    importancias = analisar_importancia_atributos(model, X_train, f'importancia_atributos_{nome_modelo}.png')
    exportar_importancia_json(importancias, f'importancia_atributos_{nome_modelo}.json')
    importance_time = time.time() - importance_start
    print(f"   ⏱️  Tempo: {importance_time:.2f}s")
    
    # 5. Visualizações de embeddings
    print("\n5. VISUALIZAÇÕES DE EMBEDDINGS")
    print("-"*40)
    embeddings_start = time.time()
    
    # PCA
    pca_start = time.time()
    visualizar_embeddings(X_test, y_test, 'pca', f'{nome_modelo}')
    pca_time = time.time() - pca_start
    print(f"   📊 PCA: {pca_time:.2f}s")
    
    # t-SNE
    tsne_start = time.time()
    visualizar_embeddings(X_test, y_test, 'tsne', f'{nome_modelo}')
    tsne_time = time.time() - tsne_start
    print(f"   📊 t-SNE: {tsne_time:.2f}s")
    
    embeddings_time = time.time() - embeddings_start
    print(f"   ⏱️  Total embeddings: {embeddings_time:.2f}s")
    
    # 6. Teste de robustez
    print("\n6. TESTE DE ROBUSTEZ")
    print("-"*40)
    robustez_start = time.time()
    resultados_robustez = teste_robustez(model, X_test, y_test, filename=f'{nome_modelo}_robustez.png')
    robustez_time = time.time() - robustez_start
    print(f"   ⏱️  Tempo: {robustez_time:.2f}s")
    
    # 7. Visualização de árvore exemplo
    print("\n7. VISUALIZAÇÃO DE ÁRVORE EXEMPLO")
    print("-"*40)
    tree_viz_start = time.time()
    model.plot_tree_graphviz(tree_idx=0, filename=f'arvore_{nome_modelo}')
    tree_viz_time = time.time() - tree_viz_start
    print(f"   ⏱️  Tempo: {tree_viz_time:.2f}s")
    
    # 8. Exportar resultados completos
    print("\n8. EXPORTAÇÃO DE RESULTADOS")
    print("-"*40)
    export_start = time.time()
    exportar_resultados_json(y_test, y_pred, model, f'resultados_{nome_modelo}.json')
    export_time = time.time() - export_start
    print(f"   ⏱️  Tempo: {export_time:.2f}s")
    
    # Tempo total do pipeline
    pipeline_time = time.time() - pipeline_start
    
    print(f"\n{'='*60}")
    print("🏁 PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    
    # Relatório detalhado de tempo
    print(f"\n⏱️  RELATÓRIO DE TEMPO DETALHADO:")
    print(f"{'─'*50}")
    print(f"   1. Treinamento:        {training_time:>8.2f}s ({training_time/pipeline_time*100:>5.1f}%)")
    print(f"   2. Predição:           {prediction_time:>8.3f}s ({prediction_time/pipeline_time*100:>5.1f}%)")
    print(f"   3. Matriz confusão:    {matrix_time:>8.2f}s ({matrix_time/pipeline_time*100:>5.1f}%)")
    print(f"   4. Importância:        {importance_time:>8.2f}s ({importance_time/pipeline_time*100:>5.1f}%)")
    print(f"   5. Embeddings:         {embeddings_time:>8.2f}s ({embeddings_time/pipeline_time*100:>5.1f}%)")
    print(f"      • PCA:              {pca_time:>8.2f}s")
    print(f"      • t-SNE:            {tsne_time:>8.2f}s")
    print(f"   6. Robustez:           {robustez_time:>8.2f}s ({robustez_time/pipeline_time*100:>5.1f}%)")
    print(f"   7. Árvore visual:      {tree_viz_time:>8.2f}s ({tree_viz_time/pipeline_time*100:>5.1f}%)")
    print(f"   8. Exportação:         {export_time:>8.2f}s ({export_time/pipeline_time*100:>5.1f}%)")
    print(f"{'─'*50}")
    print(f"   TOTAL PIPELINE:        {pipeline_time:>8.2f}s (100.0%)")
    print(f"{'='*60}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'importancia': importancias,
        'robustez': resultados_robustez,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'timing': {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'pipeline_time': pipeline_time,
            'avg_tree_time': training_time/len(model.trees),
            'predictions_per_second': len(X_test)/prediction_time
        }
    }

# Exemplo de uso:
if __name__ == "__main__":
    # Pré-processamento
    X_train, X_test, y_train, y_test = preprocess_data("random_forest/treino_sinais_vitais_com_label.txt")

    print(f"📊 Dataset Info:")
    print(f"   Features: {len(X_train.columns)}")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste: {len(X_test)} amostras")
    print(f"   Classes: {sorted(set(y_train))}")

    # CONFIGURAÇÃO: Random Forest Balanceado COM VERBOSIDADE
    model = RandomForestClassifier(
        n_estimators=200,        # Mais árvores para melhor generalização
        max_depth=8,            # Profundidade moderada (evita overfitting)
        min_samples_leaf=2,     # Menos restritivo
        max_features=None,      # Usar todos os atributos (dataset pequeno)
        bootstrap=True,         
        random_state=42,
        verbose=True            # ADICIONADO: ativar cronômetro e progresso
    )
    
    # Executar pipeline completo
    resultados = executar_pipeline_completo(model, X_train, X_test, y_train, y_test, 'rf_balanceado')
    
    print("\n🏆 RESULTADOS FINAIS:")
    print(f"   📈 Acurácia: {resultados['accuracy']:.4f}")
    print(f"   📈 F1-Score: {resultados['f1_score']:.4f}")
    print(f"   🎯 Atributo principal: {resultados['importancia']['atributo_principal']}")
    print(f"   🔬 Robustez: Degradação máxima de {max([r[2] for r in resultados['robustez']]) - min([r[2] for r in resultados['robustez']]):.4f}")
    print(f"   ⏱️  Tempo total: {resultados['timing']['pipeline_time']:.2f}s")
    print(f"   🚀 Velocidade: {resultados['timing']['predictions_per_second']:.1f} predições/segundo")