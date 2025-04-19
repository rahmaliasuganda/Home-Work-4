import heapq
import time

# Grid peta dengan elevasi dan zona terlarang (#)
grid = [
    ['S', 1, 2, 3, 'G'],
    [1, '#', 2, 3, 4],
    [2, 2, 3, '#', 5],
    [3, '#', 4, 5, 6],
    [4, 5, 6, 7, 8]
]

rows = len(grid)
cols = len(grid[0])

# Arah gerak: atas, bawah, kiri, kanan
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Cari posisi start (S) dan goal (G)
def find_positions():
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
    return start, goal

# Heuristik jarak Manhattan
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Biaya berdasarkan elevasi (jika bukan angka, default 1)
def elevation_cost(pos):
    r, c = pos
    val = grid[r][c]
    return int(val) if isinstance(val, int) else 1

# Implementasi A* Search
def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited_nodes = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_nodes += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_nodes

        for d in directions:
            nr, nc = current[0] + d[0], current[1] + d[1]
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                tentative_g = g_score[current] + elevation_cost(neighbor)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + manhattan(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    return None, visited_nodes

# Implementasi Greedy Best-First Search (GBFS)
def gbfs(start, goal):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), start))
    came_from = {}
    visited = set()
    visited_nodes = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_nodes += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_nodes

        visited.add(current)

        for d in directions:
            nr, nc = current[0] + d[0], current[1] + d[1]
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and neighbor not in visited:
                if grid[nr][nc] != '#':
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (manhattan(neighbor, goal), neighbor))
                    visited.add(neighbor)

    return None, visited_nodes

# Eksekusi utama
start, goal = find_positions()

# Ukur performa GBFS
start_time = time.time()
gbfs_path, gbfs_nodes = gbfs(start, goal)
gbfs_time = (time.time() - start_time) * 1000

# Ukur performa A*
start_time = time.time()
astar_path, astar_nodes = a_star(start, goal)
astar_time = (time.time() - start_time) * 1000

# Tampilkan hasil
print("=== Hasil Pencarian ===")
print("GBFS Path:", gbfs_path)
print(f"GBFS: {gbfs_time:.3f} ms, Nodes Visited: {gbfs_nodes}")

print("A* Path:", astar_path)
print(f"A*:   {astar_time:.3f} ms, Nodes Visited: {astar_nodes}")

# Tabel perbandingan
print("\n=== COMPARISON BASED ON TIME (ms) ===")
print(f"{'Algorithm':<10} | {'Time (ms)':>10}")
print("-" * 23)
print(f"{'GBFS':<10} | {gbfs_time:>10.3f}")
print(f"{'A*':<10} | {astar_time:>10.3f}")

print("\n=== COMPARISON BASED ON NODES VISITED ===")
print(f"{'Algorithm':<10} | {'Visited Nodes':>15}")
print("-" * 30)
print(f"{'GBFS':<10} | {gbfs_nodes:>15}")
print(f"{'A*':<10} | {astar_nodes:>15}")
