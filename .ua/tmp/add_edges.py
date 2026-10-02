import json, os

ua_dir = "/Users/leonhardaudie/Documents/antigravity/Projects/earnings-call-sentiment-analyzer/.ua"
graph_file = os.path.join(ua_dir, "knowledge-graph.json")

with open(graph_file) as f:
    graph = json.load(f)

# Define architectural dependencies (source imports target)
dependencies = [
    ("src/ecsa/pipeline.py", "src/ecsa/data/edgar.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/data/transcripts.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/data/prices.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/nlp/finbert.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/nlp/features.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/events/study.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/signals/backtest.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/settings.py"),
    
    ("src/ecsa/data/edgar.py", "src/ecsa/data/cache.py"),
    ("src/ecsa/data/edgar.py", "src/ecsa/data/ratelimit.py"),
    
    ("src/ecsa/nlp/finbert.py", "src/ecsa/nlp/cleaning.py"),
    ("src/ecsa/nlp/finbert.py", "src/ecsa/nlp/segment.py"),
    ("src/ecsa/nlp/features.py", "src/ecsa/nlp/lexicon.py"),
    ("src/ecsa/nlp/features.py", "src/ecsa/nlp/fls.py"),
    
    ("src/ecsa/events/study.py", "src/ecsa/events/market_model.py"),
    ("src/ecsa/events/study.py", "src/ecsa/events/stats.py"),
    ("src/ecsa/events/study.py", "src/ecsa/events/calendar.py"),
    
    ("src/ecsa/signals/backtest.py", "src/ecsa/signals/portfolio.py"),
    ("src/ecsa/signals/backtest.py", "src/ecsa/signals/metrics.py"),
    ("src/ecsa/signals/backtest.py", "src/ecsa/signals/model.py"),
    ("src/ecsa/signals/scoring.py", "src/ecsa/signals/backtest.py"),
    
    ("src/ecsa/dashboard.py", "src/ecsa/demo.py"),
    ("src/ecsa/demo.py", "src/ecsa/storage.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/storage.py"),
    ("src/ecsa/pipeline.py", "src/ecsa/schemas.py"),
]

edges = graph.get("edges", [])
added = 0

for source_path, target_path in dependencies:
    # Find the node IDs for these paths
    source_id = next((n["id"] for n in graph["nodes"] if n.get("filePath") == source_path and n.get("type") in ["file", "config"]), None)
    target_id = next((n["id"] for n in graph["nodes"] if n.get("filePath") == target_path and n.get("type") in ["file", "config"]), None)
    
    if source_id and target_id:
        edges.append({
            "source": source_id,
            "target": target_id,
            "type": "imports",
            "direction": "forward",
            "weight": 1.0
        })
        added += 1

graph["edges"] = edges

with open(graph_file, "w") as f:
    json.dump(graph, f)

print(f"Added {added} edges!")
