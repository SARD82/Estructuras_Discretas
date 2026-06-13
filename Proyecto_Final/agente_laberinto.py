"""
================================================================
PROYECTO FINAL - ESTRUCTURAS DISCRETAS
Facultad de Ingeniería, UNAM

Título: Agente IA Simbólico — Navegador de Laberintos con BFS

Descripción:
  Implementación de una Inteligencia Artificial Clásica/Simbólica que
  resuelve un laberinto representado como una cuadrícula. El sistema se
  divide en tres módulos que corresponden directamente a temas de la
  materia Estructuras Discretas:

    Módulo 1 — Teoría de Grafos:
      Transforma la matriz del laberinto (estructura de datos) en un
      Grafo G = (V, E) usando una lista de adyacencia (diccionario).
      Cada celda libre es un vértice V y cada par de celdas libres
      adyacentes forma una arista E.

    Módulo 2 — Lógica Proposicional:
      Define las reglas que gobiernan al agente mediante proposiciones
      lógicas combinadas con el operador de conjunción (∧).
      Un movimiento es válido sí y sólo sí se cumplen TODAS las
      proposiciones: dentro del rango + no es pared.

    Módulo 3 — Búsqueda BFS (El Agente IA):
      El agente usa Breadth-First Search (BFS), algoritmo de recorrido
      en grafos que garantiza el camino MÁS CORTO usando una Cola FIFO.
      Complejidad: O(V + E).
================================================================

DIAGRAMA DE FLUJO DEL SISTEMA (ASCII):
---------------------------------------

        ┌───────────────────────────┐
        │          INICIO           │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │       [MÓDULO 1]          │
        │  Definir laberinto como   │
        │  matriz de 0s y 1s        │
        │  (0=camino, 1=pared)      │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │       [MÓDULO 1]          │
        │  Construir Grafo G=(V,E)  │
        │  Recorrer matriz → dict   │
        │  vecinos de cada celda    │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │       [MÓDULO 3]          │
        │  Inicializar BFS:         │
        │  cola  = deque([inicio])  │
        │  visitados = {inicio}     │
        └─────────────┬─────────────┘
                      │
           ┌──────────▼──────────┐
           │    ¿Cola vacía?     │ ──SÍ──► "No hay solución" ──► FIN
           └──────────┬──────────┘
                      │ NO
                      ▼
        ┌───────────────────────────┐
        │       [MÓDULO 3]          │
        │  Sacar nodo de la cola    │
        │  (operación FIFO)         │
        └─────────────┬─────────────┘
                      │
           ┌──────────▼──────────┐
           │ ¿Nodo == Destino?   │ ──SÍ──► Reconstruir ruta ──► FIN (éxito)
           └──────────┬──────────┘
                      │ NO
                      ▼
        ┌───────────────────────────┐
        │       [MÓDULO 3]          │
        │  Para cada vecino del     │
        │  nodo actual (del grafo)  │
        └─────────────┬─────────────┘
                      │
           ┌──────────▼──────────┐
           │    [MÓDULO 2]       │
           │ ¿Movimiento válido? │ ──NO──► Ignorar vecino
           │  P ∧ Q ∧ R         │
           └──────────┬──────────┘
                      │ SÍ
           ┌──────────▼──────────┐
           │  ¿Ya fue visitado?  │ ──SÍ──► Ignorar vecino
           └──────────┬──────────┘
                      │ NO
        ┌───────────────────────────┐
        │  Agregar vecino a cola    │
        │  y marcarlo como visitado │
        └─────────────┬─────────────┘
                      │
                      └──────────────► (volver a ¿Cola vacía?)

---------------------------------------
"""

from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ================================================================
# MÓDULO 1: EL GENERADOR DEL MUNDO (GRAFOS)
# ================================================================
#
# Concepto de Estructuras Discretas aplicado:
#
#   GRAFO G = (V, E) donde:
#     V = conjunto de vértices = cada celda libre del laberinto
#     E = conjunto de aristas  = pares de celdas libres adyacentes
#
#   Representación elegida: LISTA DE ADYACENCIA (diccionario Python)
#     - Clave: tupla (fila, columna) → identifica el vértice
#     - Valor: lista de [(vecino, nombre_movimiento), ...]
#
#   Esta representación es eficiente para grafos dispersos (O(V+E) espacio),
#   lo que es típico en laberintos donde cada nodo tiene máximo 4 vecinos.
#
#   El grafo resultante es NO DIRIGIDO y SIN PESOS (las aristas no tienen
#   costo diferente, lo que hace a BFS óptimo para encontrar el camino más corto).

LABERINTO = [
    #  0  1  2  3  4  5  6   ← columnas
    [  0, 0, 1, 0, 0, 0, 0 ],  # fila 0
    [  1, 0, 1, 0, 1, 1, 0 ],  # fila 1
    [  0, 0, 0, 0, 0, 1, 0 ],  # fila 2
    [  0, 1, 1, 1, 0, 1, 0 ],  # fila 3
    [  0, 0, 0, 1, 0, 0, 0 ],  # fila 4
    [  1, 1, 0, 1, 1, 0, 1 ],  # fila 5
    [  0, 0, 0, 0, 0, 0, 0 ],  # fila 6
]
# 0 = camino libre  |  1 = pared

INICIO = (0, 0)   # vértice origen
FIN    = (6, 6)   # vértice destino

FILAS    = len(LABERINTO)
COLUMNAS = len(LABERINTO[0])

# Los 4 movimientos posibles: (Δfila, Δcol, nombre)
DIRECCIONES = [
    (-1,  0, "Arriba"),
    ( 1,  0, "Abajo"),
    ( 0, -1, "Izquierda"),
    ( 0,  1, "Derecha"),
]


def construir_grafo(laberinto):
    """
    Módulo 1 — Transforma la matriz en un grafo de adyacencia.

    Recorre cada celda libre y conecta con sus vecinas libres
    mediante el motor de reglas (Módulo 2).

    Retorna:
        grafo (dict): { (fila,col): [((fila2,col2), "movimiento"), ...] }
    """
    grafo = {}

    for fila in range(FILAS):
        for col in range(COLUMNAS):
            if laberinto[fila][col] == 0:          # solo celdas libres son vértices
                nodo = (fila, col)
                grafo[nodo] = []

                for df, dc, nombre in DIRECCIONES:
                    nf = fila + df
                    nc = col  + dc
                    if movimiento_valido(laberinto, nf, nc):   # Módulo 2
                        grafo[nodo].append(((nf, nc), nombre))

    return grafo


# ================================================================
# MÓDULO 2: EL MOTOR DE REGLAS (LÓGICA PROPOSICIONAL)
# ================================================================
#
# Concepto de Estructuras Discretas aplicado:
#
#   Se definen tres PROPOSICIONES ATÓMICAS sobre el estado del mundo:
#
#     P: "La nueva fila está dentro del rango válido [0, FILAS)"
#        → P = (0 ≤ nueva_fila < FILAS)
#
#     Q: "La nueva columna está dentro del rango válido [0, COLUMNAS)"
#        → Q = (0 ≤ nueva_col < COLUMNAS)
#
#     R: "La celda en (nueva_fila, nueva_col) no es una pared"
#        → R = (laberinto[nueva_fila][nueva_col] == 0)
#
#   La PROPOSICIÓN COMPUESTA que determina si el movimiento es válido es:
#
#       Válido(P, Q, R)  ≡  P  ∧  Q  ∧  R
#
#   Tabla de verdad parcial (el operador ∧ requiere TODOS verdaderos):
#     P=V, Q=V, R=V  →  Válido = VERDADERO   ← único caso que permite el movimiento
#     P=F, Q=V, R=V  →  Válido = FALSO       ← fuera del laberinto (fila)
#     P=V, Q=F, R=V  →  Válido = FALSO       ← fuera del laberinto (col)
#     P=V, Q=V, R=F  →  Válido = FALSO       ← es una pared
#
#   La evaluación usa CORTOCIRCUITO: si P es False, no se evalúa Q ni R
#   (optimización lógica equivalente a la conjunción).

def movimiento_valido(laberinto, fila, col):
    """
    Módulo 2 — Evalúa la proposición compuesta P ∧ Q ∧ R.

    Retorna:
        bool: True si el movimiento satisface las tres proposiciones.
    """
    P = 0 <= fila < FILAS           # proposición P: fila dentro del rango
    Q = 0 <= col  < COLUMNAS        # proposición Q: columna dentro del rango
    R = P and Q and (laberinto[fila][col] == 0)   # proposición R: no es pared

    return P and Q and R            # conjunción: P ∧ Q ∧ R


# ================================================================
# MÓDULO 3: EL AGENTE — BÚSQUEDA BFS (LA IA)
# ================================================================
#
# Concepto de Estructuras Discretas aplicado:
#
#   BFS (Breadth-First Search) es un algoritmo de RECORRIDO DE GRAFOS
#   que explora el grafo nivel por nivel (por "capas" de distancia).
#
#   Estructura de datos central: COLA (Queue) — tipo FIFO
#     First In, First Out garantiza que los nodos más cercanos al origen
#     se procesen antes, asegurando que el PRIMER camino encontrado
#     al destino sea el CAMINO MÁS CORTO (en número de aristas).
#
#   Algoritmo BFS formal:
#     1. Encolar nodo_inicio con ruta vacía []
#     2. Mientras la cola no esté vacía:
#        a. Desencolar nodo_actual y su ruta
#        b. Si nodo_actual == destino → ÉXITO, retornar ruta
#        c. Para cada vecino de nodo_actual en el grafo:
#           - Si vecino no fue visitado:
#             * Marcarlo como visitado
#             * Encolar vecino con ruta extendida
#     3. Si la cola se vació → no existe camino
#
#   Complejidad temporal:  O(V + E)
#   Complejidad espacial:  O(V)       ← el conjunto "visitados"

def bfs(grafo, inicio, fin):
    """
    Módulo 3 — El Agente IA usa BFS para encontrar el camino más corto.

    Args:
        grafo  (dict): Grafo de adyacencia construido por el Módulo 1.
        inicio (tuple): Nodo origen (fila, col).
        fin    (tuple): Nodo destino (fila, col).

    Retorna:
        list: Secuencia de movimientos ["Derecha", "Abajo", ...], o
        None: si no existe ningún camino del inicio al destino.
    """
    # Cola BFS: cada elemento es (nodo_actual, lista_de_movimientos_hasta_aquí)
    cola = deque()
    cola.append((inicio, []))

    # Conjunto de nodos ya procesados (evita ciclos y reprocesamiento)
    visitados = set()
    visitados.add(inicio)

    while cola:                                   # mientras haya nodos por explorar
        nodo_actual, ruta_actual = cola.popleft() # FIFO: sacar el más antiguo

        if nodo_actual == fin:                    # ¡destino alcanzado!
            return ruta_actual

        for vecino, movimiento in grafo.get(nodo_actual, []):
            if vecino not in visitados:           # Módulo 2 ya filtró paredes en el grafo
                visitados.add(vecino)
                cola.append((vecino, ruta_actual + [movimiento]))

    return None   # la cola se vació sin encontrar el destino


# ================================================================
# VISUALIZACIÓN: DIAGRAMA DE FLUJO EN PNG
# ================================================================

def _caja(ax, x, y, w, h, texto, color_fondo, color_texto='white', fs=8.5):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color_fondo, edgecolor='#222222', linewidth=1.8,
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y, texto, ha='center', va='center', fontsize=fs,
            color=color_texto, fontweight='bold', multialignment='center', zorder=4)


def _diamante(ax, x, y, w, h, texto, color_fondo='#F5A623', fs=8):
    hx, hy = w / 2, h / 2
    poly = plt.Polygon(
        [(x, y + hy), (x + hx, y), (x, y - hy), (x - hx, y)],
        closed=True, facecolor=color_fondo, edgecolor='#222222', linewidth=1.8, zorder=3
    )
    ax.add_patch(poly)
    ax.text(x, y, texto, ha='center', va='center', fontsize=fs,
            fontweight='bold', multialignment='center', zorder=4)


def _flecha(ax, x1, y1, x2, y2, label='', lcolor='#333333', lside='right'):
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=lcolor, lw=1.8),
                zorder=2)
    if label:
        mx = (x1 + x2) / 2 + (0.18 if lside == 'right' else -0.18)
        my = (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=8.5, color=lcolor, fontweight='bold',
                ha='center', va='center', zorder=5)


def generar_diagrama_png(ruta_salida='diagrama_flujo.png'):
    """Genera y guarda el diagrama de flujo del sistema como imagen PNG."""

    fig, ax = plt.subplots(figsize=(11, 20))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.5, 21)
    ax.axis('off')
    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    # Paleta de colores por módulo
    C_INICIO  = '#27AE60'   # verde
    C_MOD1    = '#2980B9'   # azul
    C_MOD2    = '#E67E22'   # naranja
    C_MOD3    = '#8E44AD'   # morado
    C_ERROR   = '#E74C3C'   # rojo
    C_EXITO   = '#16A085'   # verde oscuro
    C_DIAM    = '#F9CA74'   # amarillo claro (diamantes)

    CX = 5.0   # eje X del flujo principal
    W  = 3.8   # ancho cajas
    BH = 0.72  # alto cajas rectangulares
    DH = 0.95  # alto diamantes (mitad = 0.475)
    DW = W     # ancho diamantes

    # Posiciones Y (de arriba hacia abajo)
    Y = {
        'inicio':     20.0,
        'lab':        18.4,
        'grafo':      16.7,
        'bfs_init':   15.0,
        'cola':       13.2,
        'sacar':      11.4,
        'destino':     9.6,
        'vecinos':     7.8,
        'valido':      6.0,
        'visitado':    4.2,
        'agregar':     2.5,
    }

    # ── BLOQUES FLUJO PRINCIPAL ──────────────────────────────────
    _caja(ax, CX, Y['inicio'],   W, BH, "INICIO",                      C_INICIO,  fs=10)
    _caja(ax, CX, Y['lab'],      W, BH, "MÓDULO 1\nDefinir laberinto\n(matriz de 0s y 1s)", C_MOD1)
    _caja(ax, CX, Y['grafo'],    W, BH, "MÓDULO 1\nConstruir Grafo G=(V,E)\n(lista de adyacencia)", C_MOD1)
    _caja(ax, CX, Y['bfs_init'], W, BH, "MÓDULO 3 — Inicializar BFS\ncola=[inicio]  visitados={inicio}", C_MOD3)
    _diamante(ax, CX, Y['cola'],     DW, DH, "¿Cola vacía?",           C_DIAM)
    _caja(ax, CX, Y['sacar'],    W, BH, "MÓDULO 3\nSacar nodo de la cola (FIFO)", C_MOD3)
    _diamante(ax, CX, Y['destino'],  DW, DH, "¿Nodo actual == Destino?", C_DIAM)
    _caja(ax, CX, Y['vecinos'],  W, BH, "MÓDULO 3\nExplorar vecinos del nodo\n(consultar grafo)", C_MOD3)
    _diamante(ax, CX, Y['valido'],   DW, DH, "MÓDULO 2\n¿Movimiento válido?\nP ∧ Q ∧ R", C_DIAM)
    _diamante(ax, CX, Y['visitado'], DW, DH, "¿Vecino ya visitado?",   C_DIAM)
    _caja(ax, CX, Y['agregar'],  W, BH, "MÓDULO 3\nAgregar vecino a cola\ny marcar como visitado", C_MOD3)

    # ── BLOQUES LATERALES ────────────────────────────────────────
    XR = 9.3   # X columna derecha
    WR = 2.5   # ancho bloques laterales

    _caja(ax, XR, Y['cola'],     WR, BH, "Sin solución\n→ FIN", C_ERROR)
    _caja(ax, XR, Y['destino'],  WR, BH, "Reconstruir y\nmostrar ruta", C_EXITO)
    _caja(ax, XR, Y['destino'] - BH - 0.4, WR, BH, "FIN (éxito)", C_EXITO)

    # ── FLECHAS FLUJO PRINCIPAL (↓) ──────────────────────────────
    secuencia = [
        ('inicio', BH/2,  'lab',      BH/2),
        ('lab',    BH/2,  'grafo',    BH/2),
        ('grafo',  BH/2,  'bfs_init', BH/2),
        ('bfs_init',BH/2, 'cola',     DH/2),
        ('cola',   DH/2,  'sacar',    BH/2),
        ('sacar',  BH/2,  'destino',  DH/2),
        ('destino',DH/2,  'vecinos',  BH/2),
        ('vecinos',BH/2,  'valido',   DH/2),
        ('valido', DH/2,  'visitado', DH/2),
        ('visitado',DH/2, 'agregar',  BH/2),
    ]
    for k1, off1, k2, off2 in secuencia:
        _flecha(ax, CX, Y[k1] - off1, CX, Y[k2] + off2)

    # Etiquetas NO en diamantes del flujo descendente
    for key in ('cola', 'destino', 'valido', 'visitado'):
        ax.text(CX + 0.2, Y[key] - DH/2 - 0.15, 'NO', fontsize=8,
                color='#C0392B', fontweight='bold')

    # ── FLECHAS LATERALES (→ SÍ) ─────────────────────────────────
    # ¿Cola vacía? SÍ → Sin solución
    _flecha(ax, CX + DW/2, Y['cola'], XR - WR/2, Y['cola'], 'SÍ', '#C0392B')

    # ¿Es destino? SÍ → Mostrar ruta
    _flecha(ax, CX + DW/2, Y['destino'], XR - WR/2, Y['destino'], 'SÍ', '#16A085')
    _flecha(ax, XR, Y['destino'] - BH/2, XR, Y['destino'] - BH - 0.4 + BH/2)

    # ¿Movimiento válido? NO → Ignorar
    _flecha(ax, CX + DW/2, Y['valido'], 8.5, Y['valido'], 'NO', '#C0392B')
    ax.text(8.6, Y['valido'], "Ignorar\nvecino", fontsize=8, color='#7F8C8D',
            va='center', fontstyle='italic')

    # ¿Ya visitado? SÍ → Ignorar
    _flecha(ax, CX + DW/2, Y['visitado'], 8.5, Y['visitado'], 'SÍ', '#C0392B')
    ax.text(8.6, Y['visitado'], "Ignorar\nvecino", fontsize=8, color='#7F8C8D',
            va='center', fontstyle='italic')

    # ── FLECHA DE RETORNO (loop BFS) ─────────────────────────────
    lx = CX - DW/2 - 0.25   # columna izquierda del loop
    # Línea vertical desde "agregar" hasta el nivel de "¿Cola vacía?"
    ax.annotate('',
                xy=(lx, Y['cola']),
                xytext=(lx, Y['agregar']),
                arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.8), zorder=2)
    # Segmentos horizontales de conexión
    ax.plot([CX - W/2, lx], [Y['agregar'] + BH/2, Y['agregar'] + BH/2],
            color='#8E44AD', lw=1.8, zorder=2)
    ax.plot([lx, CX - DW/2], [Y['cola'], Y['cola']],
            color='#8E44AD', lw=1.8, zorder=2)
    ax.text(lx - 0.08, (Y['agregar'] + Y['cola']) / 2, "Loop\nBFS",
            fontsize=7.5, color='#8E44AD', ha='right', va='center',
            fontstyle='italic', fontweight='bold')

    # ── LEYENDA ───────────────────────────────────────────────────
    leyenda = [
        mpatches.Patch(color=C_INICIO, label='Inicio'),
        mpatches.Patch(color=C_MOD1,   label='Módulo 1 — Grafos'),
        mpatches.Patch(color=C_MOD3,   label='Módulo 3 — BFS (Agente IA)'),
        mpatches.Patch(color=C_DIAM,   label='Módulo 2 — Lógica (decisión)'),
        mpatches.Patch(color=C_EXITO,  label='Fin con éxito'),
        mpatches.Patch(color=C_ERROR,  label='Sin solución'),
    ]
    ax.legend(handles=leyenda, loc='lower left', fontsize=8.5,
              framealpha=0.95, bbox_to_anchor=(0.0, 0.0), title='Leyenda',
              title_fontsize=9)

    # ── TÍTULO ────────────────────────────────────────────────────
    ax.text(CX, 20.75,
            "Agente IA — Navegador de Laberintos\nEstructuras Discretas | FI UNAM",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1A1A2E')

    plt.tight_layout(pad=0.5)
    plt.savefig(ruta_salida, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Diagrama guardado: '{ruta_salida}'")


# ================================================================
# UTILIDADES DE IMPRESIÓN
# ================================================================

def imprimir_laberinto(laberinto, ruta_celdas=None):
    """Imprime el laberinto en consola, marcando el camino encontrado."""
    ruta_set = set(ruta_celdas) if ruta_celdas else set()
    print("\n  Laberinto  (# = pared | * = camino | I = inicio | F = fin)\n")
    for r, fila in enumerate(laberinto):
        linea = "    "
        for c, celda in enumerate(fila):
            pos = (r, c)
            if   pos == INICIO:        linea += 'I '
            elif pos == FIN:           linea += 'F '
            elif pos in ruta_set:      linea += '* '
            elif celda == 1:           linea += '# '
            else:                      linea += '  '
        print(linea)
    print()


# ================================================================
# PROGRAMA PRINCIPAL
# ================================================================

def main():
    sep = "=" * 58
    print(sep)
    print("  AGENTE IA — NAVEGADOR DE LABERINTOS")
    print("  Estructuras Discretas | FI UNAM")
    print(sep)

    # MÓDULO 1: construir el grafo
    print("\n[MÓDULO 1] Construyendo grafo del laberinto...")
    grafo = construir_grafo(LABERINTO)
    print(f"  Grafo listo: {len(grafo)} nodos (celdas libres)")
    print(f"  Inicio: {INICIO}   Destino: {FIN}")
    imprimir_laberinto(LABERINTO)

    # MÓDULO 3: ejecutar BFS
    print("[MÓDULO 3] Ejecutando BFS (búsqueda en anchura)...")
    ruta = bfs(grafo, INICIO, FIN)

    if ruta is None:
        print("\n  No existe camino del inicio al destino.\n")
    else:
        print(f"\n  Camino encontrado en {len(ruta)} pasos:")
        print(f"  {' -> '.join(ruta)}\n")

        # Reconstruir celdas del camino para visualización
        ruta_celdas = [INICIO]
        dr = {"Arriba": (-1, 0), "Abajo": (1, 0), "Izquierda": (0, -1), "Derecha": (0, 1)}
        r, c = INICIO
        for mov in ruta:
            df, dc = dr[mov]
            r += df
            c += dc
            ruta_celdas.append((r, c))

        imprimir_laberinto(LABERINTO, ruta_celdas)

    # Generar diagrama PNG
    print("[VISUAL] Generando diagrama de flujo...")
    generar_diagrama_png()

    print(sep)
    print("  Listo.")
    print(sep)


if __name__ == "__main__":
    main()
