"""图谱子图查询：返回指定实体周围的关系子图，供前端可视化。

LightRAG Neo4j 后端的存储约定：
  - 节点 label = workspace 名（如 "base"），实体名在 `entity_id`，类型在 `entity_type`
  - 边 type 统一为 "DIRECTED"，语义在 `description` / `weight` 属性里

输出格式（兼容 D3 / ECharts / Cytoscape）：
  { "nodes": [{ "id","name","label","degree","description" }],
    "edges": [{ "source","target","type","description","weight" }],
    "center": <查询的实体名> }
"""
from __future__ import annotations

from src.query import cypher_query


def _run(cypher: str, params: dict | None = None) -> list[dict]:
    """带参数的 Cypher 查询。"""
    from neo4j import GraphDatabase
    from config import settings

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [r.data() for r in result]
    finally:
        driver.close()


def get_subgraph(entity_name: str, depth: int = 1, limit: int = 50) -> dict:
    """以某实体为中心，取 depth 跳内的子图。"""
    if depth < 1 or depth > 3:
        raise ValueError("depth 取值 1-3")

    # 节点：路径上所有节点去重
    node_rows = _run(
        f"""
        MATCH p = (n)-[r*1..{depth}]-(m)
        WHERE toLower(n.entity_id) CONTAINS toLower($name)
        UNWIND nodes(p) AS node
        WITH DISTINCT node
        RETURN node.entity_id AS id,
               node.entity_type AS type,
               coalesce(node.description, '') AS description
        LIMIT $limit
        """,
        {"name": entity_name, "limit": limit},
    )

    if not node_rows:
        return {"nodes": [], "edges": [], "center": entity_name}

    node_ids = {r["id"] for r in node_rows}
    nodes = [
        {
            "id": r["id"],
            "name": r["id"],
            "label": (r.get("type") or "ENTITY"),
            "description": r.get("description", ""),
            "degree": 0,
        }
        for r in node_rows
    ]

    # 边：路径上所有关系去重，用 startNode/endNode 取端点
    edge_rows = _run(
        f"""
        MATCH p = (n)-[r*1..{depth}]-(m)
        WHERE toLower(n.entity_id) CONTAINS toLower($name)
        UNWIND relationships(p) AS rel
        WITH DISTINCT rel
        RETURN startNode(rel).entity_id AS src,
               endNode(rel).entity_id AS tgt,
               type(rel) AS type,
               coalesce(rel.description, '') AS description,
               coalesce(rel.weight, 1) AS weight
        """,
        {"name": entity_name},
    )

    edges = []
    seen = set()
    for r in edge_rows:
        src, tgt = r["src"], r["tgt"]
        if src not in node_ids or tgt not in node_ids:
            continue
        key = f"{src}->{tgt}->{r['type']}"
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source": src,
            "target": tgt,
            "type": r["type"],
            "description": r["description"],
            "weight": r["weight"],
        })

    # 度数
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1
    for n in nodes:
        n["degree"] = deg.get(n["id"], 0)

    return {"nodes": nodes, "edges": edges, "center": entity_name}


def get_top_entities(limit: int = 30) -> dict:
    """返回度数最高的实体，用于初始展示。"""
    rows = _run(
        """
        MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $limit
        RETURN n.entity_id AS name, n.entity_type AS label, degree,
               coalesce(n.description, '') AS description
        """,
        {"limit": limit},
    )
    nodes = [
        {
            "id": r["name"],
            "name": r["name"],
            "label": (r.get("label") or "ENTITY"),
            "degree": r["degree"],
            "description": r.get("description", ""),
        }
        for r in rows
        if r.get("name")
    ]
    return {"nodes": nodes, "edges": []}
