"""End-to-end fitting and one-case evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import low_rank_transport, rademacher, sample_causal_pair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def load_config(path: Path) -> Tuple[Dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Only config schema_version=1 is supported")
    # The committed config lives at <project>/configs/*.json.
    project_dir = path.parent.parent
    return config, project_dir


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config: Dict[str, Any], device: torch.device):
    model_cfg = config["model"]
    if model_cfg["dtype"] != "float32":
        raise ValueError("This minimal reproduction intentionally supports only float32")
    common = {
        "revision": model_cfg["revision"],
        "trust_remote_code": bool(model_cfg["trust_remote_code"]),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"],
        torch_dtype=torch.float32,
        attn_implementation=model_cfg["attention_implementation"],
        **common,
    )
    model.eval()
    model.config.use_cache = False
    model.to(device)
    return tokenizer, model


def tokenize(tokenizer, text: str, token_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return tokenizer(
        text,
        add_special_tokens=bool(token_cfg["add_special_tokens"]),
        max_length=int(token_cfg["max_length"]),
        truncation=bool(token_cfg["truncation"]),
        return_tensors="pt",
    )


def validate_eval_tokens(tokenizer, evaluation: Dict[str, Any]) -> Dict[str, int]:
    surfaces = [item["surface"] for item in evaluation["tracked_concepts"]]
    surfaces.append(evaluation["expected_next_token"])
    resolved: Dict[str, int] = {}
    for surface in dict.fromkeys(surfaces):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Tracked surface {surface!r} maps to {len(token_ids)} tokens {token_ids}; "
                "this vocabulary-indexed minimal evaluation requires exactly one token"
            )
        resolved[surface] = int(token_ids[0])
    return resolved


def fit_lens(
    config: Dict[str, Any],
    calibration: List[Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed = int(config["seed"])
    pair_rng = random.Random(seed)
    vector_rng = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    sample_vectors: List[torch.Tensor] = []
    per_layer_gradients: List[List[torch.Tensor]] = []
    sample_records: List[Dict[str, Any]] = []
    expected_layers = None
    expected_hidden_size = None
    n_samples = int(config["fit"]["samples_per_prompt"])

    for prompt_index, row in enumerate(calibration):
        batch = tokenize(tokenizer, row["text"], config["tokenization"])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = list(outputs.hidden_states)
        layer_states = hidden_states[:-1]
        final_state = hidden_states[-1]
        n_layers = len(layer_states)
        hidden_size = int(final_state.shape[-1])
        sequence_length = int(batch["attention_mask"].sum().item())

        if expected_layers is None:
            expected_layers = n_layers
            expected_hidden_size = hidden_size
            per_layer_gradients = [[] for _ in range(n_layers)]
        if n_layers != expected_layers or hidden_size != expected_hidden_size:
            raise RuntimeError("Model hidden-state geometry changed between prompts")

        for sample_index in range(n_samples):
            source, target = sample_causal_pair(sequence_length, pair_rng)
            vector_cpu = rademacher(hidden_size, vector_rng)
            vector = vector_cpu.to(device)
            scalar = torch.dot(final_state[0, target].float(), vector)
            # 一次 VJP 同时得到所有层的 J_l^T v；保留图以复用当前 prompt。
            gradients = torch.autograd.grad(
                scalar,
                layer_states,
                retain_graph=sample_index + 1 < n_samples,
                allow_unused=False,
            )
            sample_vectors.append(vector_cpu)
            for layer_index, gradient in enumerate(gradients):
                per_layer_gradients[layer_index].append(
                    gradient[0, source].detach().to(dtype=torch.float32, device="cpu")
                )
            sample_records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": row["id"],
                    "sample_index": sample_index,
                    "sequence_length": sequence_length,
                    "source_position": source,
                    "target_position": target,
                }
            )
        del outputs, hidden_states, layer_states, final_state

    artifact = {
        "format_version": 1,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": seed,
        "n_layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "sample_vectors": torch.stack(sample_vectors),
        "layer_gradients": torch.stack(
            [torch.stack(layer_items) for layer_items in per_layer_gradients]
        ),
    }
    summary = {
        "estimator": "mean_s v_s (J_l^T v_s)^T",
        "random_vector": "unnormalised_rademacher",
        "pair_sampling": "uniform source then uniform target in [source, length)",
        "sample_count": len(sample_records),
        "layer_count": expected_layers,
        "hidden_size": expected_hidden_size,
        "samples": sample_records,
    }
    return artifact, summary


def final_norm(model, activation: torch.Tensor) -> torch.Tensor:
    base_model = getattr(model, "model", None)
    norm = getattr(base_model, "norm", None)
    if norm is None:
        raise TypeError("Expected a Hugging Face Qwen model exposing model.norm")
    return norm(activation)


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    value = logits[token_id]
    return int(torch.count_nonzero(logits > value).item()) + 1


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    special = set(tokenizer.all_special_ids)
    # Ask for extras so filtering special tokens still normally returns k entries.
    count = min(logits.numel(), k + len(special) + 8)
    _, ids = torch.topk(logits, k=count)
    rows = []
    for token_id in ids.tolist():
        if token_id in special:
            continue
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded": tokenizer.decode([int(token_id)]),
                "logit": float(logits[token_id].item()),
            }
        )
        if len(rows) == k:
            break
    return rows


def evaluate(
    config: Dict[str, Any],
    evaluation: Dict[str, Any],
    token_ids: Dict[str, int],
    artifact: Dict[str, Any],
    tokenizer,
    model,
    device: torch.device,
) -> Dict[str, Any]:
    batch = tokenize(tokenizer, evaluation["prompt"], config["tokenization"])
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = list(outputs.hidden_states)
    layer_states = hidden_states[:-1]
    n_layers = len(layer_states)
    if n_layers != artifact["n_layers"]:
        raise RuntimeError("Lens artifact layer count does not match the loaded model")

    vectors = artifact["sample_vectors"].to(device)
    layer_gradients = artifact["layer_gradients"].to(device)
    position = int(batch["attention_mask"].sum().item()) - 1
    top_k = int(config["evaluation"]["top_k"])
    concepts = evaluation["tracked_concepts"]
    layer_rows: List[Dict[str, Any]] = []

    for layer_index, state in enumerate(layer_states):
        activation = state[0, position].float()
        transported = low_rank_transport(
            activation, vectors, layer_gradients[layer_index]
        )
        with torch.no_grad():
            j_logits = model.lm_head(final_norm(model, transported)).float()
            logit_logits = model.lm_head(final_norm(model, activation)).float()
        ranks = {}
        for concept in concepts:
            token_id = token_ids[concept["surface"]]
            ranks[concept["name"]] = {
                "surface": concept["surface"],
                "token_id": token_id,
                "j_lens_rank": token_rank(j_logits, token_id),
                "logit_lens_rank": token_rank(logit_logits, token_id),
            }
        layer_rows.append(
            {
                "layer": layer_index,
                "tracked_ranks": ranks,
                "j_lens_top": top_tokens(tokenizer, j_logits, top_k),
                "logit_lens_top": top_tokens(tokenizer, logit_logits, top_k),
            }
        )

    # The actual final state gives the model's ordinary next-token distribution.
    with torch.no_grad():
        final_logits = model.lm_head(hidden_states[-1][0, position]).float()
    answer_id = token_ids[evaluation["expected_next_token"]]
    final_answer_rank = token_rank(final_logits, answer_id)
    fractions = config["evaluation"]["middle_layer_fraction"]
    middle_start = int(n_layers * float(fractions[0]))
    middle_end = max(middle_start + 1, int(n_layers * float(fractions[1])))
    middle_rows = layer_rows[middle_start:middle_end]
    best_middle = {}
    for concept in concepts:
        name = concept["name"]
        best_middle[name] = {
            "j_lens_rank": min(row["tracked_ranks"][name]["j_lens_rank"] for row in middle_rows),
            "logit_lens_rank": min(
                row["tracked_ranks"][name]["logit_lens_rank"] for row in middle_rows
            ),
        }

    threshold = int(config["evaluation"]["answer_rank_threshold"])
    return {
        "case": evaluation,
        "prompt_token_ids": batch["input_ids"][0].tolist(),
        "analysis_position": position,
        "resolved_token_ids": token_ids,
        "middle_layer_interval": [middle_start, middle_end - 1],
        "best_middle_layer_ranks": best_middle,
        "final_answer_rank": final_answer_rank,
        "answer_rank_threshold": threshold,
        "sanity_check_passed": final_answer_rank <= threshold,
        "layers": layer_rows,
    }


def environment_record(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    dependency_names = [
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "seed": config["seed"],
        "model": config["model"],
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dependencies": {
            name: importlib.metadata.version(name) for name in dependency_names
        },
    }


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def atomic_torch_save(path: Path, value: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def render_markdown(report: Dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    env = report["environment"]
    status = "PASS" if evaluation["sanity_check_passed"] else "FAIL"
    lines = [
        "# Minimal Jacobian-lens result",
        "",
        f"- Experiment: `{env['experiment_id']}`",
        f"- Model: `{env['model']['id']}` @ `{env['model']['revision']}`",
        f"- Device: `{env['device']}`; PyTorch `{env['torch']}`; Transformers `{env['transformers']}`",
        f"- Prompt: `{evaluation['case']['prompt']}`",
        f"- Sanity check: **{status}** (answer rank {evaluation['final_answer_rank']} <= {evaluation['answer_rank_threshold']})",
        f"- Middle layers: {evaluation['middle_layer_interval'][0]}–{evaluation['middle_layer_interval'][1]} (inclusive)",
        "",
        "## Best tracked-token rank in the middle third",
        "",
        "| Concept | Surface | J-lens | Logit lens |",
        "|---|---:|---:|---:|",
    ]
    concepts = evaluation["case"]["tracked_concepts"]
    for concept in concepts:
        ranks = evaluation["best_middle_layer_ranks"][concept["name"]]
        lines.append(
            f"| {concept['name']} | `{concept['surface']}` | {ranks['j_lens_rank']} | {ranks['logit_lens_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Reading note",
            "",
            "This is an exploratory one-case method check. A high-ranked intermediate token is a lens readout, not causal evidence or proof of a global workspace. Inspect `evaluation.json` for every layer and exact top-k tokens.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    output_dir: Path, input_paths: Iterable[Path], project_dir: Path
) -> None:
    entries = []
    for path in list(input_paths) + [
        output_dir / "jlens.pt",
        output_dir / "fit_summary.json",
        output_dir / "evaluation.json",
        output_dir / "evaluation.md",
    ]:
        # Relative paths keep the checksum manifest portable across cloned locations.
        relative = path.resolve().relative_to(project_dir.resolve())
        entries.append(f"{sha256_file(path)}  {relative}")
    temp = output_dir / "manifest.sha256.tmp"
    temp.write_text("\n".join(entries) + "\n", encoding="utf-8")
    os.replace(temp, output_dir / "manifest.sha256")


def reproduce(config_path: Path, output_dir: Path, requested_device: str) -> Dict[str, Any]:
    config, project_dir = load_config(config_path)
    calibration_path = project_dir / config["data"]["calibration_path"]
    evaluation_path = project_dir / config["data"]["evaluation_path"]
    calibration = read_jsonl(calibration_path)
    evaluations = read_jsonl(evaluation_path)
    if len(evaluations) != 1:
        raise ValueError("The minimal protocol requires exactly one evaluation record")

    device = resolve_device(requested_device)
    tokenizer, model = load_model(config, device)
    token_ids = validate_eval_tokens(tokenizer, evaluations[0])
    artifact, fit_summary = fit_lens(
        config, calibration, tokenizer, model, device
    )
    evaluation = evaluate(
        config, evaluations[0], token_ids, artifact, tokenizer, model, device
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    env = environment_record(config, device)
    provenance = {
        "environment": env,
        "input_sha256": {
            "config": sha256_file(config_path),
            "calibration": sha256_file(calibration_path),
            "evaluation": sha256_file(evaluation_path),
            "dependency_lock": sha256_file(project_dir / "requirements.lock"),
        },
    }
    artifact["provenance"] = provenance
    fit_payload = {**provenance, "fit": fit_summary}
    report = {**provenance, "evaluation": evaluation}

    atomic_torch_save(output_dir / "jlens.pt", artifact)
    atomic_json(output_dir / "fit_summary.json", fit_payload)
    atomic_json(output_dir / "evaluation.json", report)
    markdown_temp = output_dir / "evaluation.md.tmp"
    markdown_temp.write_text(render_markdown(report), encoding="utf-8")
    os.replace(markdown_temp, output_dir / "evaluation.md")
    source_paths = sorted((project_dir / "src").rglob("*.py"))
    source_paths += sorted((project_dir / "scripts").glob("*.sh"))
    write_manifest(
        output_dir,
        [
            config_path,
            calibration_path,
            evaluation_path,
            project_dir / "requirements.lock",
            project_dir / "pyproject.toml",
            *source_paths,
        ],
        project_dir,
    )
    return report
