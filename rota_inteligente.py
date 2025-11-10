import os
import math
import random
import heapq
from collections import defaultdict
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------- Configurações iniciais --------------------
# Para reproduzibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Parâmetros do experimento (podem ser alterados)
NUM_NOS = 120            # número de nós do grafo (malha urbana sintética)
AREA = 100               # tamanho da área (coordenadas 0..AREA)
K_VIZINHOS = 6           # conectividade: cada nó conecta aos K_VIZINHOS mais próximos
NUM_PEDIDOS = 28         # quantidade de pedidos a serem simulados (exclui depósito)
NUM_ENTREGADORES = 4     # K para K-Means (quantas zonas / entregadores)
VELOCIDADE_KMH = 30.0    # velocidade média para estimativa de tempo (km/h)
DEPOSITO = 0             # id do nó que representa o restaurante / depósito

# Pastas de saída
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# -------------------- Utilitários do grafo --------------------
def distancia_euclidiana(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class Grafo:
    def __init__(self):
        self.adj = defaultdict(list)  # node -> list of (neighbor, peso)
        self.coords = {}              # node -> (x,y)

    def adicionar_no(self, nid, x, y):
        self.coords[nid] = (x, y)
        if nid not in self.adj:
            self.adj[nid] = []

    def adicionar_aresta(self, a, b, peso=None, bidir=True):
        if peso is None:
            peso = distancia_euclidiana(self.coords[a], self.coords[b])
        self.adj[a].append((b, peso))
        if bidir:
            self.adj[b].append((a, peso))

    def vizinhos(self, nid):
        return self.adj[nid]

    def salvar_csvs(self, path_nodes='data/nodes.csv', path_edges='data/edges.csv'):
        df_nodes = pd.DataFrame([{'id': n, 'x': xy[0], 'y': xy[1]} for n, xy in self.coords.items()])
        df_nodes.to_csv(path_nodes, index=False)
        rows = []
        seen = set()
        for a, nbrs in self.adj.items():
            for b, w in nbrs:
                if (b, a) in seen:
                    continue
                rows.append({'a': a, 'b': b, 'peso': w})
                seen.add((a, b))
        df_edges = pd.DataFrame(rows)
        df_edges.to_csv(path_edges, index=False)

# -------------------- Geração do grafo sintético --------------------
def gerar_grafo(num_nos=NUM_NOS, area=AREA, k_vizinhos=K_VIZINHOS, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    g = Grafo()
    # distribuir nós no espaço
    for i in range(num_nos):
        x = random.uniform(0, area)
        y = random.uniform(0, area)
        g.adicionar_no(i, x, y)
    # conectar cada nó aos k vizinhos mais próximos
    for i in range(num_nos):
        dists = []
        for j in range(num_nos):
            if i == j:
                continue
            dists.append((distancia_euclidiana(g.coords[i], g.coords[j]), j))
        dists.sort()
        for _, j in dists[:k_vizinhos]:
            g.adicionar_aresta(i, j)
    return g

# -------------------- A* para caminhos entre nós --------------------
def a_star(grafo, inicio, objetivo):
    if inicio == objetivo:
        return [inicio], 0.0
    open_heap = []
    heapq.heappush(open_heap, (0.0, inicio))
    came_from = {}
    g_score = {inicio: 0.0}
    closed = set()
    while open_heap:
        _, atual = heapq.heappop(open_heap)
        if atual == objetivo:
            path = [atual]
            while atual in came_from:
                atual = came_from[atual]
                path.append(atual)
            path.reverse()
            return path, g_score[objetivo]
        if atual in closed:
            continue
        closed.add(atual)
        for viz, peso in grafo.vizinhos(atual):
            tentative = g_score[atual] + peso
            if viz not in g_score or tentative < g_score[viz]:
                came_from[viz] = atual
                g_score[viz] = tentative
                est = distancia_euclidiana(grafo.coords[viz], grafo.coords[objetivo])
                f = tentative + est
                heapq.heappush(open_heap, (f, viz))
    return None, float('inf')

# -------------------- Heurística: vizinho mais próximo (TSP aproximado) --------------------
def tempo_estimado_minutos(p1, p2, velocidade_kmh=VELOCIDADE_KMH):
    # aproxima 1 grau de coordenada ~ 111 km
    dist_graus = distancia_euclidiana(p1, p2)
    dist_km = dist_graus * 111.0
    tempo_h = dist_km / velocidade_kmh
    return tempo_h * 60.0

def rota_vizinho_proximo(grafo, deposito_id, nos_cluster):
    # Mapeia ids para coordenadas
    coords = {nid: grafo.coords[nid] for nid in nos_cluster}
    rota = [deposito_id]
    pendentes = set(n for n in nos_cluster if n != deposito_id)
    total_min = 0.0
    # Construir grafo de custos (matriz implícita)
    custos = {n: {} for n in nos_cluster}
    for i in nos_cluster:
        for j in nos_cluster:
            if i == j:
                continue
            custos[i][j] = tempo_estimado_minutos(coords[i], coords[j])
    while pendentes:
        atual = rota[-1]
        melhor = None
        melhor_c = float('inf')
        for c in pendentes:
            c_v = custos[atual][c]
            if c_v < melhor_c:
                melhor_c = c_v
                melhor = c
        if melhor is None:
            break
        rota.append(melhor)
        pendentes.remove(melhor)
        total_min += melhor_c
    # volta ao depósito
    ultimo = rota[-1]
    volta = tempo_estimado_minutos(coords[ultimo], coords[deposito_id])
    rota.append(deposito_id)
    total_min += volta
    return rota, total_min

# -------------------- Funções de plotagem e salvamento --------------------
def plotar_modelo_grafo(grafo, outpath='outputs/graph_model.png'):
    plt.figure(figsize=(8,8))
    xs = [xy[0] for xy in grafo.coords.values()]
    ys = [xy[1] for xy in grafo.coords.values()]
    plt.scatter(xs, ys, s=10, alpha=0.6)
    # desenha arestas (uma direção por par)
    seen = set()
    for a, nbrs in grafo.adj.items():
        for b, w in nbrs:
            if (b,a) in seen:
                continue
            xa, ya = grafo.coords[a]
            xb, yb = grafo.coords[b]
            plt.plot([xa, xb], [ya, yb], linewidth=0.5, alpha=0.3)
            seen.add((a,b))
    plt.title('Modelo do Grafo (nós e arestas)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plotar_clusters(grafo, entregas_ids, rotulos, k, deposito_id=DEPOSITO, outpath='outputs/clusters.png'):
    plt.figure(figsize=(10,8))
    xs = [grafo.coords[n][0] for n in grafo.coords]
    ys = [grafo.coords[n][1] for n in grafo.coords]
    plt.scatter(xs, ys, s=8, alpha=0.2)
    cores = plt.colormaps.get_cmap('tab10').resampled(max(1, k))
    for i, eid in enumerate(entregas_ids):
        x,y = grafo.coords[eid]
        plt.scatter(x, y, s=90, color=cores(rotulos[i] % 10), edgecolor='k')
    dx, dy = grafo.coords[deposito_id]
    plt.scatter(dx, dy, s=200, marker='*', color='red', label='Depósito')
    plt.title(f'Clusters de Entregas (K={k})')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plotar_rotas(grafo, rotas_info, outpath='outputs/routes.png'):
    plt.figure(figsize=(10,8))
    xs = [grafo.coords[n][0] for n in grafo.coords]
    ys = [grafo.coords[n][1] for n in grafo.coords]
    plt.scatter(xs, ys, s=8, alpha=0.2)
    cores = plt.colormaps.get_cmap('tab10').resampled(max(1, len(rotas_info)))
    for idx, (rota, mins) in enumerate(rotas_info):
        rx = [grafo.coords[n][0] for n in rota]
        ry = [grafo.coords[n][1] for n in rota]
        plt.plot(rx, ry, '-', linewidth=2, label=f'Rota {idx} ({mins:.1f} min)')
        plt.scatter(rx, ry, s=40, color=cores(idx%10))
    plt.title('Rotas por Cluster (heurística)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------- Pipeline principal --------------------
def executar_pipeline():
    t0 = time.time()
    print('Gerando grafo sintético...')
    grafo = gerar_grafo()
    grafo.salvar_csvs()
    print('Grafo gerado e salvo em data/.')

    # gerar pedidos aleatórios (exclui depósito)
    nos_possiveis = list(grafo.coords.keys())
    if DEPOSITO in nos_possiveis:
        nos_possiveis.remove(DEPOSITO)
    pedidos = random.sample(nos_possiveis, min(NUM_PEDIDOS, len(nos_possiveis)))
    df_pedidos = pd.DataFrame([{'id': pid, 'x': grafo.coords[pid][0], 'y': grafo.coords[pid][1]} for pid in pedidos])
    df_pedidos.to_csv('data/deliveries.csv', index=False)
    print('Pedidos salvos em data/deliveries.csv')

    # clustering com K-Means (usamos as coordenadas dos pedidos)
    pontos = np.array([[grafo.coords[p][0], grafo.coords[p][1]] for p in pedidos])
    kmeans = KMeans(n_clusters=NUM_ENTREGADORES, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(pontos)
    plotar_modelo_grafo(grafo)
    plotar_clusters(grafo, pedidos, labels, NUM_ENTREGADORES)

    # construir rotas por cluster
    clusters = defaultdict(list)
    for p, lab in zip(pedidos, labels):
        clusters[lab].append(p)

    rotas_info = []
    resumo = []
    for cid, nos_cluster in clusters.items():
        # incluir depósito no conjunto para rota_vizinho_proximo
        conjunto = [DEPOSITO] + nos_cluster
        print(f'Construindo rota para cluster {cid} (n={len(nos_cluster)})...')
        rota, mins = rota_vizinho_proximo(grafo, DEPOSITO, conjunto)
        rotas_info.append((rota, mins))
        resumo.append({'cluster': cid, 'n_pontos': len(nos_cluster), 'tempo_min': mins})

    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv('outputs/routes_summary.csv', index=False)
    plotar_rotas(grafo, rotas_info)

    t1 = time.time()
    print('Pipeline concluído. Tempo total: {:.2f}s'.format(t1 - t0))
    print('Arquivos gerados: data/ , outputs/')
    return grafo, pedidos, labels, rotas_info

if __name__ == '__main__':
    executar_pipeline()
