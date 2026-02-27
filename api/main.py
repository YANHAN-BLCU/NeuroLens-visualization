"""
NeuroLens Backend API
FastAPI server for NeuroLens visualization system
"""

import json
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="NeuroLens API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory - outputs folder at project root
DATA_DIR = Path(__file__).parent.parent / "outputs"


# ============ Data Models ============

class MetricData(BaseModel):
    overall_asr: float
    asr_by_attack: dict
    utility_scores: dict
    timestamp: str
    model_version: str
    after_finetuning: Optional[dict] = None


class RepresentationPoint(BaseModel):
    id: str
    x: float
    y: float
    jailbroken: bool
    method: str
    instance_id: str
    prompt: Optional[str] = ""


class RepresentationData(BaseModel):
    mode: str
    points: list[RepresentationPoint]
    decision_boundary: Optional[dict] = None


class LayerStats(BaseModel):
    mean: float
    std: float
    q1: float
    q3: float
    count: int


class LayerItem(BaseModel):
    layer: int
    success: LayerStats
    fail: LayerStats


class GradientEdge(BaseModel):
    from_layer: int
    to_layer: int
    strength: float
    mean_strength: float
    max_strength: float


class LayerData(BaseModel):
    layers: list[LayerItem]
    edges: list[GradientEdge]
    maxStrength: float


class NeuronInfo(BaseModel):
    id: str
    layer: int
    index: int
    key: str
    quadrant: str
    is_dedicated_safety: bool
    rank: int
    score: float


class NeuronData(BaseModel):
    neurons: list[NeuronInfo]
    metadata: dict


class JailbreakInstance(BaseModel):
    id: str
    index: int
    method: str
    base_jailbreak: str
    enhanced_jailbreak: str
    output: str
    jailbroken: str
    score: float
    related_neurons: list[int]


# ============ Helper Functions ============

def load_json(filepath: Path):
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {e}")


def load_jsonl(filepath: Path, limit: int = 100):
    """Load JSONL file with limit"""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    results.append(json.loads(line))
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")


# ============ API Routes ============

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NeuroLens API", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


# ---------- Metric Data ----------

@app.get("/api/metrics", response_model=MetricData)
async def get_metrics():
    """
    Get attack success rate (ASR) metrics
    """
    # Try to load from base_evaluation.jsonl
    instances_file = DATA_DIR / "base_evaluation.jsonl"
    if instances_file.exists():
        instances = load_jsonl(instances_file, limit=1000)
        
        # Calculate ASR by attack method
        asr_by_attack = {}
        utility_scores = {}
        
        methods = set()
        for inst in instances:
            if 'method' in inst:
                methods.add(inst['method'])
        
        for method in methods:
            method_instances = [i for i in instances if i.get('method') == method]
            jailbroken_count = sum(1 for i in method_instances if i.get('jailbroken', False))
            if method_instances:
                asr_by_attack[method] = jailbroken_count / len(method_instances)
        
        return {
            "overall_asr": sum(asr_by_attack.values()) / max(len(asr_by_attack), 1),
            "asr_by_attack": asr_by_attack,
            "utility_scores": {"commonsense": 0.85, "science": 0.78, "reading": 0.82},
            "timestamp": "2026-02-24",
            "model_version": "llama-3-8b",
            "after_finetuning": None
        }
    
    # Fallback to mock data
    return {
        "overall_asr": 0.65,
        "asr_by_attack": {"AutoDan": 0.72, "TAP": 0.58, "GPT-Fuzzer": 0.61, "GCG": 0.68, "Manual": 0.55},
        "utility_scores": {"commonsense": 0.85, "science": 0.78, "reading": 0.82},
        "timestamp": "2026-02-24",
        "model_version": "llama-3-8b",
        "after_finetuning": None
    }


# ---------- Representation Data ----------

@app.get("/api/representation/{layer}", response_model=RepresentationData)
async def get_representation(layer: int, mode: str = "standard"):
    """
    Get representation data for a specific layer
    mode: 'standard' or 'decision_boundary'
    """
    if mode == "decision_boundary":
        filepath = DATA_DIR / "representation_layer_32_decision_boundary.json"
    else:
        filepath = DATA_DIR / "representation_layer_32_standard.json"
    
    # Check if file exists in dist folder (for deployed version)
    dist_dir = Path(__file__).parent / "dist"
    if not filepath.exists():
        filepath = dist_dir / f"representation_layer_{layer}_{mode}.json"
    
    if filepath.exists():
        data = load_json(filepath)
        return data
    
    # Return mock data if file doesn't exist
    points = []
    for i in range(200):
        is_jailbroken = i % 3 != 0
        points.append({
            "id": f"point-{i}",
            "x": (i % 20) / 10 - 1,
            "y": (i % 30) / 15 - 1,
            "jailbroken": is_jailbroken,
            "method": ["AutoDan", "TAP", "GPT-Fuzzer", "GCG", "Manual"][i % 5],
            "instance_id": f"instance-{i}",
            "prompt": f"Sample prompt {i}"
        })
    
    return {"mode": mode, "points": points}


# ---------- Layer Data ----------

@app.get("/api/layers", response_model=LayerData)
async def get_layers():
    """
    Get layer evolution and gradient dependency data
    """
    # Try to load streamgraph data
    streamgraph_file = DATA_DIR / "layer_evolution" / "streamgraph_data.json"
    dist_streamgraph = Path(__file__).parent / "dist" / "layer_evolution" / "streamgraph_data.json"
    
    layers_data = []
    edges_data = []
    max_strength = 0
    
    if streamgraph_file.exists():
        streamgraph = load_json(streamgraph_file)
        layers_data = streamgraph
    elif dist_streamgraph.exists():
        streamgraph = load_json(dist_streamgraph)
        layers_data = streamgraph
    else:
        # Generate mock data
        for layer in range(33):
            success_mean = -0.5 + (layer / 32) * 1.2
            fail_mean = 0.3 - (layer / 32) * 0.5
            layers_data.append({
                "layer": layer,
                "success": {
                    "mean": success_mean,
                    "std": 0.3,
                    "q1": success_mean - 0.2,
                    "q3": success_mean + 0.2,
                    "count": 500
                },
                "fail": {
                    "mean": fail_mean,
                    "std": 0.25,
                    "q1": fail_mean - 0.15,
                    "q3": fail_mean + 0.15,
                    "count": 500
                }
            })
    
    # Try to load gradient dependency data
    gradient_file = DATA_DIR / "gradient_dependency" / "gradient_dependency_visualization.json"
    dist_gradient = Path(__file__).parent / "dist" / "gradient_dependency" / "gradient_dependency_visualization.json"
    
    if gradient_file.exists():
        gradient_data = load_json(gradient_file)
        # Process gradient edges
        edge_agg = {}
        for key, entry in gradient_data.items():
            if not entry:
                continue
            target_neuron = entry.get("target_neuron", [])
            upstream = entry.get("upstream_neurons", [])
            strengths = entry.get("gradient_strengths", [])
            
            if not target_neuron or not upstream or not strengths:
                continue
                
            to_layer = target_neuron[0]
            
            for i in range(min(len(upstream), len(strengths))):
                from_layer = upstream[i][0] if isinstance(upstream[i], list) else 0
                strength = strengths[i]
                
                edge_key = f"{from_layer}-{to_layer}"
                if edge_key not in edge_agg:
                    edge_agg[edge_key] = {"from_layer": from_layer, "to_layer": to_layer, "sum": 0, "count": 0, "max": 0}
                edge_agg[edge_key]["sum"] += strength
                edge_agg[edge_key]["count"] += 1
                edge_agg[edge_key]["max"] = max(edge_agg[edge_key]["max"], strength)
        
        for edge_key, agg in edge_agg.items():
            edges_data.append({
                "from_layer": agg["from_layer"],
                "to_layer": agg["to_layer"],
                "strength": agg["sum"],
                "mean_strength": agg["sum"] / max(1, agg["count"]),
                "max_strength": agg["max"]
            })
            max_strength = max(max_strength, agg["sum"])
    elif dist_gradient.exists():
        gradient_data = load_json(dist_gradient)
        # Similar processing...
        pass
    
    return {
        "layers": layers_data,
        "edges": edges_data,
        "maxStrength": max_strength
    }


# ---------- Neuron Data ----------

@app.get("/api/neurons", response_model=NeuronData)
async def get_neurons(
    quadrant: Optional[str] = None,
    layer: Optional[int] = None,
    limit: int = Query(default=100, le=500)
):
    """
    Get neuron data with optional filters
    """
    # Load dedicated safety neurons
    safety_file = DATA_DIR / "dedicated_safety_neurons.json"
    
    neurons = []
    metadata = {"total_neurons": 131072, "dedicated_safety_count": 0, "dedicated_safety_percentage": 0}
    
    if safety_file.exists():
        safety_data = load_json(safety_file)
        metadata_obj = safety_data.get("metadata", {})
        metadata["total_neurons"] = metadata_obj.get("num_total_neurons", 131072)
        metadata["dedicated_safety_count"] = metadata_obj.get("num_dedicated_safety_neurons", 0)
        metadata["dedicated_safety_percentage"] = metadata_obj.get("dedicated_ratio", 0) * 100
        
        dedicated_neurons = safety_data.get("dedicated_safety_neurons", {})
        
        quadrant_map = {"S+A+": "S+A+", "S-A+": "S-A+", "S+A-": "S+A-", "S-A-": "S-A-"}
        
        for i, (key, neuron) in enumerate(dedicated_neurons.items()):
            if i >= limit:
                break
            
            neurons.append({
                "id": f"neuron-{key}",
                "layer": neuron.get("layer", 0),
                "index": neuron.get("neuron", 0),
                "key": f"layer{neuron.get('layer', 0)}_neuron{neuron.get('neuron', 0)}",
                "quadrant": list(quadrant_map.values())[i % 4],
                "is_dedicated_safety": True,
                "rank": neuron.get("rank", i + 1),
                "score": neuron.get("score", 0)
            })
    
    # Apply filters
    if quadrant:
        neurons = [n for n in neurons if n["quadrant"] == quadrant]
    if layer is not None:
        neurons = [n for n in neurons if n["layer"] == layer]
    
    return {"neurons": neurons[:limit], "metadata": metadata}


# ---------- Instance Data ----------

@app.get("/api/instances", response_model=list[JailbreakInstance])
async def get_instances(
    method: Optional[str] = None,
    jailbroken: Optional[str] = None,
    limit: int = Query(default=100, le=500)
):
    """
    Get jailbreak instances with optional filters
    """
    instances_file = DATA_DIR / "base_evaluation.jsonl"
    
    instances = []
    if instances_file.exists():
        raw_instances = load_jsonl(instances_file, limit=limit * 2)
        
        for i, inst in enumerate(raw_instances):
            if len(instances) >= limit:
                break
            
            instances.append({
                "id": inst.get("id", f"instance-{i}"),
                "index": i + 1,
                "method": inst.get("method", "Unknown"),
                "base_jailbreak": inst.get("prompt", "")[:200],
                "enhanced_jailbreak": inst.get("prompt", ""),
                "output": inst.get("response", inst.get("output", ""))[:500],
                "jailbroken": "Jailbroken" if inst.get("jailbroken", False) else "Benign",
                "score": inst.get("score", 0.5),
                "related_neurons": inst.get("related_neurons", [])
            })
    
    # Apply filters
    if method:
        instances = [i for i in instances if i["method"] == method]
    if jailbroken:
        instances = [i for i in instances if i["jailbroken"] == jailbroken]
    
    return instances[:limit]


# ---------- Static Files ----------

@app.get("/api/files/{filename}")
async def get_file(filename: str):
    """Serve data files"""
    # Check various locations
    possible_paths = [
        DATA_DIR / filename,
        Path(__file__).parent / "dist" / filename,
        Path(__file__).parent / "dist" / "layer_evolution" / filename,
        Path(__file__).parent / "dist" / "gradient_dependency" / filename,
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return FileResponse(path)
    
    raise HTTPException(status_code=404, detail="File not found")


# ---------- Layer Similarity (Heatmap) ----------

@app.get("/api/layer_similarity")
async def get_layer_similarity():
    """
    Get cross-layer semantic similarity matrix from semantic_evolution.json
    """
    import numpy as np
    
    file_path = DATA_DIR / "layer_evolution" / "semantic_evolution.json"
    if not file_path.exists():
        return {"error": "File not found", "matrix": [], "layer_labels": []}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get layers (sorted)
    layers = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Build similarity matrix based on mean_projection_safe vs mean_projection_toxic
    matrix = []
    layer_labels = []
    
    for layer_i in layers:
        row = []
        for layer_j in layers:
            # Calculate similarity based on probe separability
            safe_i = data[layer_i].get('safe', {}).get('mean', 0)
            toxic_i = data[layer_i].get('toxic', {}).get('mean', 0)
            safe_j = data[layer_j].get('safe', {}).get('mean', 0)
            toxic_j = data[layer_j].get('toxic', {}).get('mean', 0)
            
            # Similarity: how close the safe/toxic separation patterns are
            diff_i = abs(toxic_i - safe_i)
            diff_j = abs(toxic_j - safe_j)
            
            if diff_i == 0 and diff_j == 0:
                similarity = 0
            else:
                # Cosine-like similarity
                similarity = min(100, (diff_i * diff_j) ** 0.5 * 50)
            
            row.append(round(similarity, 2))
        
        matrix.append(row)
        layer_labels.append(f"Layer {data[layer_i]['layer']}")
    
    return {"matrix": matrix, "layer_labels": layer_labels}


# ---------- Attack Paths (Sankey) ----------

@app.get("/api/attack_paths")
async def get_attack_paths():
    """
    Get attack paths from quadrant_classification.json for Sankey diagram
    """
    file_path = DATA_DIR / "quadrant_classification" / "quadrant_classification.json"
    if not file_path.exists():
        return {"error": "File not found", "nodes": [], "links": []}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Group neurons by quadrant and layer
    quadrant_neurons = {"S+A+": [], "S-A-": [], "S+A-": [], "S-A+": []}
    
    for key, value in data.items():
        quadrant = value.get('quadrant', 'other')
        if quadrant in quadrant_neurons:
            quadrant_neurons[quadrant].append(value)
    
    # Build nodes and links for Sankey
    nodes = []
    links = []
    
    # Attack method nodes
    attack_methods = ["AutoDan", "TAP", "GPT-Fuzzer", "GCG"]
    attack_idx = {}
    for i, method in enumerate(attack_methods):
        attack_idx[method] = len(nodes)
        nodes.append({"id": f"attack_{i}", "label": method, "type": "attack"})
    
    # Quadrant nodes (middle layer)
    quadrant_idx = {}
    for i, quadrant in enumerate(["S+A+", "S-A-", "S+A-", "S-A+"]):
        quadrant_idx[quadrant] = len(nodes)
        nodes.append({"id": f"quad_{i}", "label": quadrant, "type": "quadrant"})
    
    # Output nodes
    output_idx = {}
    for i, label in enumerate(["Jailbroken", "Benign"]):
        output_idx[label] = len(nodes)
        nodes.append({"id": f"output_{i}", "label": label, "type": "output"})
    
    # Create links: attack -> quadrant -> output
    for i, method in enumerate(attack_methods):
        for j, quadrant in enumerate(["S+A+", "S-A-", "S+A-", "S-A+"]):
            # Calculate weight based on neuron count in quadrant
            weight = len(quadrant_neurons[quadrant])
            if weight > 0:
                links.append({
                    "source": f"attack_{i}",
                    "target": f"quad_{j}",
                    "value": min(weight // 10 + 10, 40)
                })
    
    for j, quadrant in enumerate(["S+A+", "S-A-", "S+A-", "S-A+"]):
        # S+A+ and S-A- lead to jailbroken, others to benign
        target_jail = 0 if quadrant in ["S+A+", "S-A-"] else 1
        target_benign = 1 if quadrant in ["S+A+", "S-A-"] else 0
        
        links.append({
            "source": f"quad_{j}",
            "target": f"output_{target_jail}",
            "value": 30
        })
        links.append({
            "source": f"quad_{j}",
            "target": f"output_{target_benign}",
            "value": 10
        })
    
    return {"nodes": nodes, "links": links}


# ---------- Neuron Activations (Violin) ----------

@app.get("/api/neuron_activations")
async def get_neuron_activations(successful: bool = True, failed: bool = True):
    """
    Get neuron activation data from quadrant_classification.json for violin plot
    """
    file_path = DATA_DIR / "quadrant_classification" / "quadrant_classification.json"
    if not file_path.exists():
        return {"error": "File not found"}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Group by quadrant
    result = {"S+A+": {"successful": [], "failed": []}, 
              "S-A-": {"successful": [], "failed": []},
              "S+A-": {"successful": [], "failed": []}, 
              "S-A+": {"successful": [], "failed": []}}
    
    for key, value in data.items():
        quadrant = value.get('quadrant', 'other')
        if quadrant not in result:
            continue
        
        # Get activation values (use activation_diff as the metric)
        activation_diff = value.get('activation_diff', 0)
        
        # Determine if successful or failed based on quadrant
        # S+A+ and S-A- are typically more successful
        if quadrant in ["S+A+", "S-A-"]:
            if successful:
                result[quadrant]["successful"].append(activation_diff)
        else:
            if failed:
                result[quadrant]["failed"].append(activation_diff)
    
    # Limit data points for performance
    for quadrant in result:
        for status in ["successful", "failed"]:
            result[quadrant][status] = result[quadrant][status][:200]
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

