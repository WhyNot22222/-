from collections import deque

class PetriNodeType:
    Place = "Place"
    Transition = "Transition"

class PetriNode:
    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type

class PetriNet:
    def __init__(self, nodes, edges, S, EN, PT, RC, RL, RT):
        self.nodes = nodes      # 节点，可能是库所或变迁
        self.edges = edges      # 边，(库所, 变迁), (变迁, 库所)
        self.S = S              # 开始节点
        self.EN = EN            # 结束节点
        self.PT = PT            # 变迁触发的概率
        self.RC = RC            # 库所（组件）的可靠度
        self.RL = RL            # 连接件的可靠度
        self.RT = RT            # 变迁的可靠度

        self.g = [[] for _ in range(len(nodes))]    # 邻接表
        self.name2idx = {node.name: i for i, node in enumerate(nodes)}  # 节点名到索引的映射

        for u, v in edges:
            self.g[self.name2idx[u]].append(self.name2idx[v])   # 构建邻接表

    def get_paths(self) -> list[list[PetriNode]]:
        ans = []
        start_node = self.nodes[self.name2idx[self.S[0]]]
        que = deque([(start_node, [start_node])])

        while que:
            node, path = que.popleft()
            if node.name in self.EN:    # 到达终结状态
                ans.append(path)
                continue

            for idx in self.g[self.name2idx[node.name]]:    # 遍历其相邻节点
                next_node = self.nodes[idx]
                if path.count(next_node) >= 2:      # 允许出现一次回路
                    continue
                new_path = path + [next_node]
                que.append((next_node, new_path))

        return ans

    def get_path_probability(self, path) -> float:
        ans = 1.0
        for node in filter(lambda n: n.node_type == PetriNodeType.Transition, path):
            ans *= self.PT[node.name]
        return ans

    def get_path_reliability(self, path) -> float:
        ans = 1.0

        for node in filter(lambda n: n.name not in self.S and n.name not in self.EN, path):
            if node.node_type in PetriNodeType.Place:
                ans *= self.RC[node.name.replace('P', 'C')]
            else:
                ans *= self.RT[node.name]
                ans *= self.RL[node.name.replace('T', 'L')]

        return ans

    def get_system_reliability(self, path_probabilities, path_reliabilities) -> float:
        assert len(path_probabilities) == len(path_reliabilities)
        ans = sum(p * r for p, r in zip(path_probabilities, path_reliabilities))
        ans /= sum(path_probabilities)
        return ans


if __name__ == '__main__':
    nodes = [
        PetriNode("S", PetriNodeType.Place),
        PetriNode("P1", PetriNodeType.Place),
        PetriNode("P2", PetriNodeType.Place),
        PetriNode("P3", PetriNodeType.Place),
        PetriNode("P4", PetriNodeType.Place),
        PetriNode("P5", PetriNodeType.Place),
        PetriNode("P6", PetriNodeType.Place),
        PetriNode("P7", PetriNodeType.Place),
        PetriNode("P8", PetriNodeType.Place),
        PetriNode("P9", PetriNodeType.Place),
        PetriNode("EN", PetriNodeType.Place),
        PetriNode("T1", PetriNodeType.Transition),
        PetriNode("T2", PetriNodeType.Transition),
        PetriNode("T3", PetriNodeType.Transition),
        PetriNode("T4", PetriNodeType.Transition),
        PetriNode("T5", PetriNodeType.Transition),
        PetriNode("T6", PetriNodeType.Transition),
        PetriNode("T7", PetriNodeType.Transition),
        PetriNode("T8", PetriNodeType.Transition),
        PetriNode("T9", PetriNodeType.Transition),
        PetriNode("T10", PetriNodeType.Transition)
    ]

    edges = [
        ("S", "P1"), ("P1", "T1"), ("T1", "P2"), ("P2", "T2"), ("T2", "P3"),
        ("P3", "T3"), ("T3", "P4"), ("P4", "T4"), ("T4", "P5"), ("P5", "T5"),
        ("T5", "P6"), ("P6", "T6"), ("T6", "P7"), ("P7", "T7"), ("T7", "P8"), ("P8", "T8"),
        ("T8", "P2"), ("P4", "T10"), ("T10", "P9"), ("P9", "EN"), ("P7", "T9"),
        ("T9", "P9"),
    ]

    PT = {
        "T1": 1, "T2": 0.99, "T3": 0.98, "T4": 0.80, "T5": 1.0,
        "T6": 1.0, "T7": 0.30, "T8": 0.98, "T9": 0.98, "T10": 0.20
    }

    RC = {
        "C1": 1, "C2": 0.99, "C3": 0.98, "C4": 1, "C5": 0.99,
        "C6": 0.99, "C7": 1, "C8": 0.98, "C9": 1,
    }

    RL = {
        "L1": 0.99, "L2": 1, "L3": 1, "L4": 0.98, "L5": 1,
        "L6": 0.99, "L7": 0.99, "L8": 1, "L9": 0.98, "L10": 1
    }

    RT = {
        "T1": 1, "T2": 0.99, "T3": 1, "T4": 0.98, "T5": 0.99,
        "T6": 1, "T7": 0.98, "T8": 0.98, "T9": 0.99, "T10": 1
    }

    pnet = PetriNet(nodes, edges, ["S"], ["EN"], PT, RC, RL, RT)
    paths = pnet.get_paths()
    path_probabilities = []
    path_reliabilities = []

    for i, path in enumerate(paths):
        path_probabilities.append(pnet.get_path_probability(path))
        path_reliabilities.append(pnet.get_path_reliability(path))

        print(f"路径 {i} : {' -> '.join(node.name.replace('P', 'C') for node in path)}")
        print(f"迁移概率 : {path_probabilities[-1]:.3f}")
        print(f"可靠度 : {path_reliabilities[-1]:.3f}\n")

    print(f"系统可靠度 : {pnet.get_system_reliability(path_probabilities, path_reliabilities)}")
