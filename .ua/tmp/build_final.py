import json, os

ua_dir = "/Users/leonhardaudie/Documents/antigravity/Projects/earnings-call-sentiment-analyzer/.ua"
with open(os.path.join(ua_dir, "intermediate/scan-result.json")) as f:
    scan = json.load(f)
with open(os.path.join(ua_dir, "intermediate/assembled-graph.json")) as f:
    graph = json.load(f)
with open(os.path.join(ua_dir, "intermediate/layers.json")) as f:
    layers = json.load(f)
with open(os.path.join(ua_dir, "intermediate/tour.json")) as f:
    tour = json.load(f)

final = {
  "version": "1.0.0",
  "project": {
    "name": scan.get("name", "earnings-call-sentiment-analyzer"),
    "languages": scan.get("languages", ["python"]),
    "frameworks": scan.get("frameworks", []),
    "description": scan.get("description", "Analyzes sentiment"),
    "analyzedAt": "2026-10-02T00:00:00Z",
    "gitCommitHash": "none"
  },
  "nodes": graph.get("nodes", []),
  "edges": graph.get("edges", []),
  "layers": layers,
  "tour": tour
}

with open(os.path.join(ua_dir, "knowledge-graph.json"), "w") as f:
    json.dump(final, f)
