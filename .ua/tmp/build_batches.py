import json, os, subprocess

base_dir = "/Users/leonhardaudie/Documents/antigravity/Projects/earnings-call-sentiment-analyzer"
plugin_root = "/Users/leonhardaudie/Documents/antigravity/Projects/Understand-Anything/understand-anything-plugin"
ua_dir = os.path.join(base_dir, ".ua")

with open(os.path.join(ua_dir, "intermediate/batches.json")) as f:
    batches_data = json.load(f)

for batch in batches_data.get("batches", []):
    idx = batch["batchIndex"]
    
    input_data = {
        "projectRoot": base_dir,
        "batchFiles": batch["files"],
        "batchImportData": batch.get("batchImportData", {})
    }
    
    in_file = os.path.join(ua_dir, f"tmp/input-{idx}.json")
    out_file = os.path.join(ua_dir, f"tmp/out-{idx}.json")
    
    with open(in_file, "w") as f:
        json.dump(input_data, f)
        
    cmd = ["node", os.path.join(plugin_root, "skills/understand/extract-structure.mjs"), in_file, out_file]
    subprocess.run(cmd, check=True)
    
    with open(out_file) as f:
        ast_result = json.load(f)
        
    nodes = []
    edges = []
    
    for res in ast_result.get("results", []):
        file_path = res["path"]
        cat = next((f["fileCategory"] for f in batch["files"] if f["path"] == file_path), "code")
        file_type = cat if cat in ["file", "config", "document"] else "file"
        
        file_node_id = f"{file_type}:{file_path}"
        nodes.append({
            "id": file_node_id,
            "type": file_type,
            "name": os.path.basename(file_path),
            "summary": f"Generated summary for {os.path.basename(file_path)}",
            "tags": ["auto"],
            "complexity": "simple",
            "filePath": file_path
        })
        
        for func in res.get("functions", []):
            name = func.get("name") if isinstance(func, dict) else func
            func_id = f"function:{file_path}:{name}"
            nodes.append({
                "id": func_id,
                "type": "function",
                "name": name,
                "summary": f"Function {name}",
                "tags": ["auto"],
                "complexity": "simple",
                "filePath": file_path,
                "lineRange": func.get("lineRange", [0, 0]) if isinstance(func, dict) else [0, 0]
            })
            edges.append({
                "source": file_node_id,
                "target": func_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
            
        for cls in res.get("classes", []):
            name = cls.get("name") if isinstance(cls, dict) else cls
            cls_id = f"class:{file_path}:{name}"
            nodes.append({
                "id": cls_id,
                "type": "class",
                "name": name,
                "summary": f"Class {name}",
                "tags": ["auto"],
                "complexity": "simple",
                "filePath": file_path,
                "lineRange": cls.get("lineRange", [0, 0]) if isinstance(cls, dict) else [0, 0]
            })
            edges.append({
                "source": file_node_id,
                "target": cls_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
                
    batch_out = os.path.join(ua_dir, f"intermediate/batch-{idx}.json")
    with open(batch_out, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f)
        
print("Batches built successfully.")
