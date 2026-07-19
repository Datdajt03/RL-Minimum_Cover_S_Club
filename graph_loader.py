"""
graph_loader.py
Đọc các file đồ thị từ pbodulieu sử dụng igraph:
  - .gml  → igraph GML format
  - .clq  → DIMACS clique format (p edge N M / e u v)
"""
import os
import re
import igraph as ig


def load_gml(filepath: str) -> ig.Graph:
    try:
        # Thử đọc trực tiếp bằng API của igraph
        G = ig.Graph.Read_GML(filepath)
    except Exception:
        # Fallback: Một số file GML có định dạng đặc thù hoặc lặp cạnh
        # Đọc thủ công và parse bằng Regex
        with open(filepath, 'r') as f:
            content = f.read()
        
        nodes = re.findall(r'id\s+(\d+)', content)
        node_ids = sorted(list(set(int(n) for n in nodes)))
        n_nodes = len(node_ids)
        
        # Tạo ánh xạ từ ID gốc sang index 0-indexed của igraph
        id_map = {orig_id: idx for idx, orig_id in enumerate(node_ids)}
        
        edges = re.findall(r'source\s+(\d+)\s+target\s+(\d+)', content)
        edges_list = []
        for u, v in edges:
            edges_list.append((id_map[int(u)], id_map[int(v)]))
            
        G = ig.Graph(n=n_nodes, edges=edges_list, directed=False)
        G.simplify()  # Bỏ qua cạnh lặp và khuyên tự nối
    
    return G


def load_clq(filepath: str) -> ig.Graph:
    edges = []
    n_nodes = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                # p edge N M
                n_nodes = int(parts[2])
            elif parts[0] == 'e':
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # Chuyển sang 0-indexed
                edges.append((u, v))
                
    G = ig.Graph(n=n_nodes, edges=edges, directed=False)
    G.simplify()
    return G


def load_graph(filepath: str) -> ig.Graph:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.gml':
        return load_gml(filepath)
    elif ext == '.clq':
        return load_clq(filepath)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {ext}")


def load_all_graphs(data_dir: str) -> dict:
    """Đọc tất cả file đồ thị trong thư mục, trả về dict {tên: Graph}"""
    graphs = {}
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in ('.gml', '.clq'):
            try:
                G = load_graph(fpath)
                name = os.path.splitext(fname)[0]
                graphs[name] = G
                print(f"  Loaded {name}: {G.vcount()} nodes, {G.ecount()} edges")
            except Exception as ex:
                print(f"  ERROR loading {fname}: {ex}")
    return graphs
