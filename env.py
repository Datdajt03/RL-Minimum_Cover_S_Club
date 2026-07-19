"""
env.py — SClubEnvironment đúng theo paper EVO-RL sử dụng thư viện igraph

State vector 5 chiều theo paper (Phương trình state):
  z_t = [f_min(P), f_avg(P), rho_feas(P), phi_bar(P), tau]

  f_min  = fitness tốt nhất trong quần thể (nhỏ = tốt)
  f_avg  = fitness trung bình quần thể
  rho_feas = tỷ lệ cá thể khả thi trong quần thể
  phi_bar  = vi phạm đường kính trung bình
  tau     = số bước không có cải tiến (stagnation counter)

Reward 4 trường hợp theo paper (Phương trình reward):
  +α₁  nếu số club giảm
  +α₂  nếu tỷ lệ khả thi tăng (mà không giảm club)
  -α₃  nếu vi phạm tăng
  -α₄  nếu không có tiến bộ nào

3 toán tử:
  0 = structural_crossover     — trao đổi nhãn club giữa 2 cha mẹ
  1 = feasibility_preserving   — di chuyển đỉnh giữa club gần nhau
  2 = diversification          — xáo trộn nhiều phân công hơn
"""
import numpy as np
import igraph as ig
from experiment_config import ExperimentConfig

# Reward coefficients theo paper (Phương pháp đề xuất - Bảng 1):
# α₁=10 (club giảm), α₂=5 (feasibility tăng), α₃=5 (vi phạm tăng), α₄=1 (không tiến bộ)
ALPHA1 = 10.0   # club count giảm → phần thưởng cao nhất
ALPHA2 = 5.0    # feasibility ratio tăng
ALPHA3 = 5.0    # vi phạm tăng → phạt
ALPHA4 = 1.0    # không có tiến bộ → phạt nhỏ


def get_subgraph_diameter(graph: ig.Graph, nodes: list, s: int) -> int:
    """Tính đường kính của subgraph tạo bởi các đỉnh 'nodes' bằng igraph C++."""
    if len(nodes) <= 1:
        return 0
    subg = graph.subgraph(nodes)
    if not subg.is_connected():
        return -1  # Không liên thông
    return subg.diameter()


class Individual:
    """Một cá thể trong quần thể — đại diện bằng assignment vector."""
    def __init__(self, n_nodes, k_max):
        self.n = n_nodes
        self.k_max = k_max
        self._x = np.random.randint(0, k_max, size=n_nodes)
        self.invalidate_cache()

    def invalidate_cache(self):
        self._cached_phi = None
        self._cached_fitness = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.invalidate_cache()

    def num_clubs(self):
        return len(set(self._x))

    def fitness(self, graph: ig.Graph, s, penalty_lambda=10.0):
        if self._cached_fitness is None:
            k = self.num_clubs()
            phi = self._diameter_violation(graph, s)
            self._cached_fitness = k + penalty_lambda * phi
        return self._cached_fitness

    def is_feasible(self, graph: ig.Graph, s):
        return self._diameter_violation(graph, s) == 0

    def _diameter_violation(self, graph: ig.Graph, s):
        if self._cached_phi is None:
            total = 0
            for club_id in set(self._x):
                nodes_in = [v for v in range(self.n) if self._x[v] == club_id]
                if len(nodes_in) <= 1:
                    continue
                d = get_subgraph_diameter(graph, nodes_in, s)
                if d == -1:
                    total += len(nodes_in)  # Phạt không liên thông
                elif d > s:
                    total += (d - s)
            self._cached_phi = total
        return self._cached_phi


class SClubEnvironment:
    """
    Môi trường tiến hóa cho bài toán Minimum s-Club Cover.
    State = state vector 5 chiều theo paper.
    Action = chọn 1 trong 3 toán tử tiến hóa.
    """
    def __init__(self, graph: ig.Graph = None, s: int = 2, pop_size: int = 10):
        config = ExperimentConfig()
        self.s = s
        self.pop_size = pop_size
        self.penalty_lambda = config.penalty_lambda
        self.max_generations = config.max_generations
        self.stagnation_limit = config.stagnation_limit

        # Đồ thị
        if graph is None:
            # Đồ thị vòng 10 đỉnh mặc định
            self.graph = ig.Graph.Ring(10)
        else:
            self.graph = graph

        self.nodes = list(range(self.graph.vcount()))
        self.n = len(self.nodes)
        self.k_max = max(2, self.n // 2)
        
        # Danh sách kề tối ưu hóa từ igraph
        self.adj = self.graph.get_adjlist()

        # Quần thể và trạng thái
        self.population = []
        self.best_feasible = None
        self.best_fitness = float('inf')
        self.tau = 0            # stagnation counter
        self.prev_rho_feas = 0.0
        self.prev_phi_bar  = 0.0
        self.prev_k_best   = self.k_max
        self.step_count    = 0

    def reset(self):
        """Khởi tạo quần thể ngẫu nhiên tham lam."""
        self.population = [Individual(self.n, self.k_max) for _ in range(self.pop_size)]
        for ind in self.population:
            self._local_repair(ind)
        self.best_feasible = None
        self.best_fitness  = float('inf')
        self.tau           = 0
        self.step_count    = 0
        self._update_stats()
        return self._get_state()

    def step(self, action):
        """
        action ∈ {0, 1, 2} — chọn toán tử:
          0 = structural crossover
          1 = feasibility-preserving mutation
          2 = diversification mutation
        """
        self.step_count += 1
        prev_state = self._get_stats_snapshot()

        # Áp dụng toán tử
        if action == 0:
            self._structural_crossover()
        elif action == 1:
            self._feasibility_preserving_mutation()
        else:
            self._diversification_mutation()

        # Repair + selection
        self._repair_and_select()
        self._update_stats()

        reward = self._compute_reward(prev_state)
        done   = (self.step_count >= self.max_generations) or (
            self.best_feasible is not None and
            self.best_feasible.num_clubs() <= 1) or (
            self.tau >= self.stagnation_limit)
        state  = self._get_state()
        return state, reward, done, {}

    # ── State & Stats ──────────────────────────────────────────────────────

    def _update_stats(self):
        fitnesses = [ind.fitness(self.graph, self.s, self.penalty_lambda)
                     for ind in self.population]
        feasible  = [ind for ind in self.population if ind.is_feasible(self.graph, self.s)]
        self.f_min     = min(fitnesses)
        self.f_avg     = float(np.mean(fitnesses))
        self.rho_feas  = len(feasible) / self.pop_size
        violations     = [ind._diameter_violation(self.graph, self.s) for ind in self.population]
        self.phi_bar   = float(np.mean(violations))

        if feasible:
            best_f = min(feasible, key=lambda x: x.num_clubs())
            if self.best_feasible is None or best_f.num_clubs() < self.prev_k_best:
                self.best_feasible = best_f
                self.prev_k_best   = best_f.num_clubs()
                self.tau = 0
            else:
                self.tau += 1
        else:
            self.tau += 1

    def _get_state(self):
        """State vector 5 chiều theo paper."""
        return np.array([
            self.f_min,
            self.f_avg,
            self.rho_feas,
            self.phi_bar,
            float(self.tau),
        ], dtype=np.float32)

    def _get_stats_snapshot(self):
        return {
            'k_best':   self.prev_k_best,
            'rho_feas': self.rho_feas,
            'phi_bar':  self.phi_bar,
        }

    # ── Reward theo paper ──────────────────────────────────────────────────

    def _compute_reward(self, prev):
        k_now   = self.prev_k_best
        rho_now = self.rho_feas
        phi_now = self.phi_bar

        if k_now < prev['k_best']:
            return ALPHA1                         # club giảm → tốt nhất
        if rho_now > prev['rho_feas'] and k_now == prev['k_best']:
            return ALPHA2                         # khả thi tăng
        if phi_now > prev['phi_bar']:
            return -ALPHA3                        # vi phạm tăng
        return -ALPHA4                            # không tiến bộ

    # ── Operators ─────────────────────────────────────────────────────────

    def _structural_crossover(self):
        """Trao đổi nhãn club giữa 2 cha mẹ ngẫu nhiên."""
        if len(self.population) < 2:
            return
        idx_a, idx_b = np.random.choice(len(self.population), 2, replace=False)
        pa, pb = self.population[idx_a], self.population[idx_b]
        child = Individual(self.n, self.k_max)
        mask = np.random.rand(self.n) > 0.5
        child.x = np.where(mask, pa.x, pb.x)
        self.population.append(child)

    def _feasibility_preserving_mutation(self):
        """Di chuyển một đỉnh sang club lân cận, kiểm tra ràng buộc đường kính."""
        parent = self.population[np.random.randint(len(self.population))]
        child  = Individual(self.n, self.k_max)
        child.x = parent.x.copy()
        v = np.random.randint(self.n)
        neighbors = self.graph.neighbors(v)
        if neighbors:
            u = neighbors[np.random.randint(len(neighbors))]
            child.x[v] = child.x[u]   # gán v vào club của hàng xóm
            child.invalidate_cache()
        self.population.append(child)

    def _diversification_mutation(self):
        """Xáo trộn nhiều đỉnh hơn để thoát local optima."""
        parent = self.population[np.random.randint(len(self.population))]
        child  = Individual(self.n, self.k_max)
        child.x = parent.x.copy()
        n_perturb = max(1, self.n // 5)
        idx = np.random.choice(self.n, n_perturb, replace=False)
        child.x[idx] = np.random.randint(0, self.k_max, size=n_perturb)
        child.invalidate_cache()
        self.population.append(child)

    def _repair_and_select(self):
        """
        Repair: chỉ di chuyển đỉnh vi phạm đường kính sang club tốt hơn cho cá thể con mới.
        Selection: giữ pop_size cá thể tốt nhất (tournament).
        """
        if len(self.population) > self.pop_size:
            self._local_repair(self.population[-1])

        # Sort và giữ pop_size cá thể
        self.population.sort(
            key=lambda x: x.fitness(self.graph, self.s, self.penalty_lambda))
        self.population = self.population[:self.pop_size]

    def _local_repair(self, ind):
        """Di chuyển đỉnh vi phạm sang club ít vi phạm hơn."""
        for club_id in set(ind.x.copy()):
            nodes_in = [v for v in range(self.n) if ind.x[v] == club_id]
            if len(nodes_in) <= 1:
                continue
            d = get_subgraph_diameter(self.graph, nodes_in, self.s)
            if d == -1 or d > self.s:
                # Tính degree của các đỉnh chỉ trong subgraph
                nodes_set = set(nodes_in)
                subg_degrees = {v: sum(1 for nbr in self.adj[v] if nbr in nodes_set) for v in nodes_in}
                # Tránh trường hợp worst không có kết nối trong subgraph (sẽ bị cô lập)
                worst = min(nodes_in, key=lambda v: subg_degrees[v])
                
                # Chuyển worst sang club của một neighbor của nó nếu có
                neighbors = self.adj[worst]
                if neighbors:
                    new_club = ind.x[neighbors[np.random.randint(len(neighbors))]]
                else:
                    new_club = (club_id + 1) % self.k_max
                
                ind.x[worst] = new_club
                ind.invalidate_cache()

    # ── Compat helpers ────────────────────────────────────────────────────

    def get_cover_size(self):
        if self.best_feasible is not None:
            return self.best_feasible.num_clubs()
        return self.prev_k_best

    def is_fully_covered(self):
        if self.best_feasible is None:
            return False
        return set(range(self.n)) == set(
            v for v in range(self.n)  # tất cả đỉnh đều được assign
        )