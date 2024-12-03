import graphviz

class FaultTreeNode:
    def __init__(self, name, node_type="basic", children=None):
        """
        初始化故障树节点

        参数:
        name (str): 节点名称
        node_type (str): 节点类型，可选值为 "basic"（基本事件）、"intermediate"（中间事件）、"top"（顶事件），默认为 "basic"
        children (list): 子节点列表，默认为None
        """
        self.name = name
        self.node_type = node_type
        self.children = children if children else []

    def add_child(self, child):
        """
        添加子节点到当前节点

        参数:
        child (FaultTreeNode): 要添加的子节点对象
        """
        self.children.append(child)

def visualize_fault_tree(root):
    """
    使用graphviz可视化故障树

    参数:
    root (FaultTreeNode): 故障树的根节点
    """
    dot = graphviz.Digraph()
    _add_nodes_and_edges(dot, root)
    return dot

def _add_nodes_and_edges(dot, node):
    """
    递归地添加节点和边到graphviz的Digraph对象

    参数:
    dot (graphviz.Digraph): graphviz的Digraph对象
    node (FaultTreeNode): 当前要处理的故障树节点
    """
    dot.node(node.name, node.name)
    for child in node.children:
        dot.node(child.name, child.name)
        dot.edge(node.name, child.name)
        _add_nodes_and_edges(dot, child)

def qualitative_analysis(root):
    """
    对故障树进行定性分析，求最小割集

    参数:
    root (FaultTreeNode): 故障树的根节点

    返回:
    list: 最小割集列表，每个最小割集是一个节点名称的列表
    """
    if root.node_type == "basic":
        return [[root.name]]
    cut_sets = []
    for child in root.children:
        child_cut_sets = qualitative_analysis(child)
        if root.node_type == "AND":
            # 对于与门，将子节点的割集对应元素合并（笛卡尔积形式）
            new_cut_sets = []
            if cut_sets:
                for cs1 in cut_sets:
                    for cs2 in child_cut_sets:
                        new_cut_sets.append(cs1 + cs2)
            else:
                new_cut_sets = child_cut_sets
            cut_sets = new_cut_sets
        elif root.node_type == "OR":
            # 对于或门，直接合并子节点的割集
            cut_sets.extend(child_cut_sets)
    # 简化割集（去除包含其他割集的割集来得到最小割集，简单实现示例，可优化性能）
    minimal_cut_sets = []
    for cut_set in cut_sets:
        is_minimal = True
        for existing_minimal in minimal_cut_sets:
            if all(item in cut_set for item in existing_minimal):
                is_minimal = False
                break
        if is_minimal:
            minimal_cut_sets.append(cut_set)
    return minimal_cut_sets

def quantitative_analysis(root, basic_event_probs):
    """
    对故障树进行定量分析，计算顶事件发生概率

    参数:
    root (FaultTreeNode): 故障树的根节点
    basic_event_probs (dict): 基本事件名称到其发生概率的映射字典

    返回:
    float: 顶事件发生概率
    """
    minimal_cut_sets = qualitative_analysis(root)
    top_event_prob = 0
    for cut_set in minimal_cut_sets:
        cut_set_prob = 1
        for event_name in cut_set:
            cut_set_prob *= basic_event_probs[event_name]
        top_event_prob += cut_set_prob
    return top_event_prob

# 构建一个简单故障树示例（这里是一个或门下面接两个与门，与门再各自接两个基本事件的简单结构）
top_event = FaultTreeNode("Top Event", node_type="OR")
intermediate1 = FaultTreeNode("Intermediate 1", node_type="AND")
intermediate2 = FaultTreeNode("Intermediate 2", node_type="OR")
basic1 = FaultTreeNode("Basic 1")
basic2 = FaultTreeNode("Basic 2")
basic3 = FaultTreeNode("Basic 3")
basic4 = FaultTreeNode("Basic 4")

intermediate1.add_child(basic1)
intermediate1.add_child(basic2)
intermediate2.add_child(basic3)
intermediate2.add_child(basic4)

top_event.add_child(intermediate1)
top_event.add_child(intermediate2)

# 可视化故障树
dot = visualize_fault_tree(top_event)
dot.render('fault_tree', view=True)

# 进行定性分析求最小割集
minimal_cut_sets = qualitative_analysis(top_event)
print("最小割集:", minimal_cut_sets)

# 假设基本事件概率（这里简单赋值示例）
basic_event_probs = {
    "Basic 1": 0.1,
    "Basic 2": 0.2,
    "Basic 3": 0.3,
    "Basic 4": 0.4
}
# 进行定量分析计算顶事件概率
top_event_prob = quantitative_analysis(top_event, basic_event_probs)
print("顶事件发生概率:", top_event_prob)