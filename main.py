from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()


SAFE_INT_MAX = 9007199254740991

INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

CHOOSE_CODES = [
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
]

REPAIR_CODES = [
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
]


def is_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def is_positive_safe_int(x):
    return is_safe_int(x) and x > 0


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def utf8_sorted_unique(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def round12(x):
    return round(float(x), 12)


# ---------------------------------------------------------
# CHOOSE
# ---------------------------------------------------------

def choose_operation(data):
    reason_codes = {name: [] for name in INTERVENTIONS}
    total_costs = {}

    policy = data.get("policy")
    candidates = data.get("candidates")

    valid_policy = isinstance(policy, dict)
    valid_candidates = (
        isinstance(candidates, list)
        and len(candidates) == 4
        and all(isinstance(c, dict) for c in candidates)
    )

    if not valid_policy or not valid_candidates:
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {},
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    candidate_map = {}

    for c in candidates:
        name = c.get("name")

        if name in INTERVENTIONS and name not in candidate_map:
            candidate_map[name] = c

    # Exactly one candidate for each intervention
    if set(candidate_map.keys()) != set(INTERVENTIONS):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {
                name: 0 for name in INTERVENTIONS
            },
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    # Validate policy
    required_policy = [
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    ]

    policy_valid = all(k in policy for k in required_policy)

    if policy_valid:
        policy_valid = (
            finite_number(policy["minQuality"])
            and 0 <= policy["minQuality"] <= 1
            and isinstance(policy["freshnessRequired"], bool)
            and finite_number(policy["maxLatencyMs"])
            and policy["maxLatencyMs"] >= 0
            and finite_number(policy["maxMemoryMb"])
            and policy["maxMemoryMb"] >= 0
            and is_safe_int(policy["maxLabeledExamples"])
            and finite_number(policy["maxTotalCost"])
            and policy["maxTotalCost"] >= 0
            and is_safe_int(policy["horizonRequests"])
        )

    if not policy_valid:
        for name in INTERVENTIONS:
            reason_codes[name].append("INVALID_INPUT")

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {
                name: 0 for name in INTERVENTIONS
            },
            "reasonCodes": {
                name: utf8_sorted_unique(reason_codes[name])
                for name in INTERVENTIONS
            },
        }

    eligible = []

    for name in INTERVENTIONS:
        c = candidate_map[name]

        valid = True

        required_candidate = [
            "available",
            "quality",
            "freshness",
            "latencyMs",
            "memoryMb",
            "labeledExamples",
            "oneTimeCost",
            "recurringCost",
        ]

        if (
            any(k not in c for k in required_candidate)
            or not isinstance(c.get("available"), bool)
            or not finite_number(c.get("quality"))
            or not 0 <= c.get("quality") <= 1
            or not isinstance(c.get("freshness"), bool)
            or not finite_number(c.get("latencyMs"))
            or c.get("latencyMs") < 0
            or not finite_number(c.get("memoryMb"))
            or c.get("memoryMb") < 0
            or not is_safe_int(c.get("labeledExamples"))
            or not finite_number(c.get("oneTimeCost"))
            or c.get("oneTimeCost") < 0
            or not finite_number(c.get("recurringCost"))
            or c.get("recurringCost") < 0
        ):
            reason_codes[name].append("INVALID_INPUT")
            total_costs[name] = 0
            continue

        cost = round12(
            c["oneTimeCost"]
            + policy["horizonRequests"] * c["recurringCost"]
        )

        total_costs[name] = cost

        if not c["available"]:
            reason_codes[name].append("UNAVAILABLE")
            valid = False

        if c["quality"] < policy["minQuality"]:
            reason_codes[name].append("QUALITY_FLOOR")
            valid = False

        if policy["freshnessRequired"] and not c["freshness"]:
            reason_codes[name].append("FRESHNESS_REQUIRED")
            valid = False

        if c["latencyMs"] > policy["maxLatencyMs"]:
            reason_codes[name].append("LATENCY_LIMIT")
            valid = False

        if c["memoryMb"] > policy["maxMemoryMb"]:
            reason_codes[name].append("MEMORY_LIMIT")
            valid = False

        if c["labeledExamples"] > policy["maxLabeledExamples"]:
            reason_codes[name].append("DATA_LIMIT")
            valid = False

        if cost > policy["maxTotalCost"]:
            reason_codes[name].append("COST_LIMIT")
            valid = False

        if valid:
            eligible.append(name)

    for name in INTERVENTIONS:
        reason_codes[name] = utf8_sorted_unique(reason_codes[name])

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }


# ---------------------------------------------------------
# REPAIR
# ---------------------------------------------------------

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def repair_operation(data):
    reason_codes = []

    tokens = data.get("tokens")

    labels = []

    token_valid = isinstance(tokens, list) and len(tokens) > 0

    if token_valid:
        for token in tokens:
            valid_token = (
                isinstance(token, dict)
                and is_safe_int(token.get("id"))
                and token.get("role") in {
                    "system",
                    "user",
                    "assistant",
                }
                and isinstance(token.get("padding"), bool)
                and isinstance(token.get("text"), str)
            )

            if not valid_token:
                token_valid = False
                break

    if token_valid:
        labels = [
            token["id"]
            if token["role"] == "assistant" and not token["padding"]
            else -100
            for token in tokens
        ]
    else:
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []
        reason_codes.append("INVALID_TOKEN")

    # Template
    template_pass = data.get("templateApplications") == 1

    if not template_pass:
        reason_codes.append("CHAT_TEMPLATE_COUNT")

    # Parameters
    parameters = data.get("parameters")
    allowed_targets = data.get("allowedTargets")

    parameter_pass = (
        isinstance(parameters, list)
        and isinstance(allowed_targets, list)
        and len(allowed_targets) > 0
        and all(isinstance(x, str) and x != "" for x in allowed_targets)
        and len(set(allowed_targets)) == len(allowed_targets)
    )

    names_seen = set()
    trainable = []

    if parameter_pass:
        for p in parameters:
            valid_p = (
                isinstance(p, dict)
                and isinstance(p.get("name"), str)
                and p["name"] != ""
                and isinstance(p.get("target"), str)
                and p["target"] != ""
                and is_positive_safe_int(p.get("numel"))
                and p["name"] not in names_seen
            )

            if not valid_p:
                parameter_pass = False
                break

            names_seen.add(p["name"])

            if (
                p["target"] in allowed_targets
                and (
                    p["name"].endswith(".lora_A.weight")
                    or p["name"].endswith(".lora_B.weight")
                )
            ):
                trainable.append(p)

    if not parameter_pass:
        reason_codes.append("INVALID_PARAMETER")

    if parameter_pass and not trainable:
        reason_codes.append("INVALID_PARAMETER")
        parameter_pass = False

    trainable_names = sorted(
        [p["name"] for p in trainable],
        key=lambda x: x.encode("utf-8"),
    )

    trainable_count = 0
    for p in trainable:
        trainable_count += p["numel"]

    # PEFT / inference
    inference_pass = data.get("inferenceMode") is False

    if not inference_pass:
        reason_codes.append("INFERENCE_MODE")

    peft_config_pass = parameter_pass and inference_pass

    # Artifact files
    artifact_files = data.get("artifactFiles")

    expected_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    adapter_files = []

    if (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and sorted(artifact_files, key=lambda x: x.encode("utf-8"))
        == sorted(expected_files, key=lambda x: x.encode("utf-8"))
    ):
        adapter_files = sorted(
            artifact_files,
            key=lambda x: x.encode("utf-8"),
        )
    else:
        reason_codes.append("ADAPTER_FILE_SET")

    # If any full model artifact is supplied
    if isinstance(artifact_files, list):
        full_model_names = {
            "pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.safetensors",
        }

        if any(x in full_model_names for x in artifact_files):
            reason_codes.append("FULL_MODEL_ARTIFACT")

    # Checkpoint
    checkpoint = data.get("checkpoint")

    required_checkpoint = {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition",
    }

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and required_checkpoint.issubset(checkpoint.keys())
    )

    if not checkpoint_complete:
        reason_codes.append("INCOMPLETE_CHECKPOINT")

    # Lineage
    base_revision = data.get("baseRevision")
    dataset_digest = data.get("datasetDigest")
    code_digest = data.get("codeDigest")
    config_digest = data.get("configDigest")
    expected_digests = data.get("expectedDigests")

    lineage_pass = True

    if not isinstance(base_revision, str) or not HEX40.fullmatch(base_revision):
        reason_codes.append("MUTABLE_BASE_REVISION")
        lineage_pass = False

    for digest in [
        dataset_digest,
        code_digest,
        config_digest,
    ]:
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            lineage_pass = False

    if not lineage_pass and "MUTABLE_BASE_REVISION" not in reason_codes:
        reason_codes.append("LINEAGE_MISMATCH")

    if lineage_pass:
        if not isinstance(expected_digests, dict):
            lineage_pass = False
        else:
            for key, value in [
                ("datasetDigest", dataset_digest),
                ("codeDigest", code_digest),
                ("configDigest", config_digest),
            ]:
                if expected_digests.get(key) != value:
                    lineage_pass = False

            if not lineage_pass:
                reason_codes.append("LINEAGE_MISMATCH")

    # Batch
    micro = data.get("microBatch")
    accumulation = data.get("gradientAccumulation")
    replicas = data.get("replicas")
    expected_batch = data.get("expectedEffectiveBatch")

    batch_pass = (
        is_positive_safe_int(micro)
        and is_positive_safe_int(accumulation)
        and is_positive_safe_int(replicas)
        and is_positive_safe_int(expected_batch)
        and micro * accumulation * replicas == expected_batch
    )

    if not batch_pass:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")

    # Evaluation isolation
    train_ids = data.get("trainRowIds")
    eval_ids = data.get("evalRowIds")

    ids_valid = (
        isinstance(train_ids, list)
        and isinstance(eval_ids, list)
        and len(train_ids) > 0
        and len(eval_ids) > 0
        and all(isinstance(x, str) and x != "" for x in train_ids)
        and all(isinstance(x, str) and x != "" for x in eval_ids)
        and len(set(train_ids)) == len(train_ids)
        and len(set(eval_ids)) == len(eval_ids)
    )

    eval_isolated = ids_valid and set(train_ids).isdisjoint(set(eval_ids))

    if ids_valid and not eval_isolated:
        reason_codes.append("EVAL_LEAKAGE")
    elif not ids_valid:
        reason_codes.append("EVAL_LEAKAGE")

    eval_dropout = data.get("dropoutActiveDuringEval") is False

    if not eval_dropout:
        reason_codes.append("EVAL_DROPOUT_ACTIVE")

    # Resume
    uninterrupted = data.get("uninterruptedWeights")
    resumed = data.get("resumedWeights")
    tolerance = data.get("resumeTolerance")

    resume_pass = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted) == len(resumed)
        and all(finite_number(x) for x in uninterrupted)
        and all(finite_number(x) for x in resumed)
        and finite_number(tolerance)
        and tolerance >= 0
    )

    if resume_pass:
        for a, b in zip(uninterrupted, resumed):
            if abs(float(a) - float(b)) > float(tolerance):
                resume_pass = False
                break

    if not resume_pass:
        reason_codes.append("RESUME_DIVERGENCE")

    evaluation_deterministic = (
        eval_dropout
        and resume_pass
    )

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": utf8_sorted_unique(reason_codes),
    }


# ---------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------

@app.post("/adapt")
async def adapt(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    operation = data.get("operation")

    if operation not in {"choose", "repair"}:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if operation == "choose":
        return choose_operation(data)

    return repair_operation(data)


@app.get("/")
def root():
    return {"status": "ok"}
