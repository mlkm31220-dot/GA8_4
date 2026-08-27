from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Any
import math
import re

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

INTERVENTIONS = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
]

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

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# =========================================================
# HELPERS
# =========================================================

def is_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_positive_safe_int(value):
    return is_safe_int(value) and value > 0


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def utf8_sort(values):
    return sorted(values, key=lambda x: x.encode("utf-8"))


def utf8_sort_unique(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def rounded_cost(value):
    return round(float(value), 12)


# =========================================================
# CHOOSE OPERATION
# =========================================================

def choose_operation(data):

    reason_codes = {
        name: []
        for name in INTERVENTIONS
    }

    total_costs = {}

    policy = data.get("policy")
    candidates = data.get("candidates")

    # -----------------------------------------------------
    # Basic structure
    # -----------------------------------------------------

    if not isinstance(policy, dict):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {},
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTIONS
            },
        }

    if not isinstance(candidates, list) or len(candidates) != 4:
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {},
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTIONS
            },
        }

    # -----------------------------------------------------
    # Policy validation
    # -----------------------------------------------------

    required_policy = [
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    ]

    policy_valid = all(
        key in policy
        for key in required_policy
    )

    if policy_valid:

        policy_valid = (
            finite_number(policy["minQuality"])
            and 0 <= float(policy["minQuality"]) <= 1

            and isinstance(
                policy["freshnessRequired"],
                bool
            )

            and finite_number(
                policy["maxLatencyMs"]
            )
            and policy["maxLatencyMs"] >= 0

            and finite_number(
                policy["maxMemoryMb"]
            )
            and policy["maxMemoryMb"] >= 0

            and is_safe_int(
                policy["maxLabeledExamples"]
            )

            and finite_number(
                policy["maxTotalCost"]
            )
            and policy["maxTotalCost"] >= 0

            and is_safe_int(
                policy["horizonRequests"]
            )
        )

    if not policy_valid:
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {},
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTIONS
            },
        }

    # -----------------------------------------------------
    # Candidate map
    # -----------------------------------------------------

    candidate_map = {}

    for candidate in candidates:

        if not isinstance(candidate, dict):
            continue

        name = candidate.get("name")

        if (
            name in INTERVENTIONS
            and name not in candidate_map
        ):
            candidate_map[name] = candidate

    if set(candidate_map.keys()) != set(INTERVENTIONS):

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {},
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTIONS
            },
        }

    # -----------------------------------------------------
    # Evaluate candidates
    # -----------------------------------------------------

    eligible = []

    required_candidate = [
        "name",
        "available",
        "quality",
        "freshness",
        "latencyMs",
        "memoryMb",
        "labeledExamples",
        "oneTimeCost",
        "recurringCost",
    ]

    for name in INTERVENTIONS:

        candidate = candidate_map[name]

        valid_candidate = all(
            key in candidate
            for key in required_candidate
        )

        if not valid_candidate:
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        # Basic candidate validation
        if not isinstance(
            candidate["available"],
            bool
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if (
            not finite_number(candidate["quality"])
            or not 0 <= candidate["quality"] <= 1
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if not isinstance(
            candidate["freshness"],
            bool
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if (
            not finite_number(candidate["latencyMs"])
            or candidate["latencyMs"] < 0
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if (
            not finite_number(candidate["memoryMb"])
            or candidate["memoryMb"] < 0
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if not is_safe_int(
            candidate["labeledExamples"]
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if (
            not finite_number(
                candidate["oneTimeCost"]
            )
            or candidate["oneTimeCost"] < 0
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        if (
            not finite_number(
                candidate["recurringCost"]
            )
            or candidate["recurringCost"] < 0
        ):
            reason_codes[name].append(
                "INVALID_INPUT"
            )
            continue

        # -------------------------------------------------
        # Cost
        # -------------------------------------------------

        cost = rounded_cost(
            candidate["oneTimeCost"]
            + (
                policy["horizonRequests"]
                * candidate["recurringCost"]
            )
        )

        total_costs[name] = cost

        passes = True

        # Availability
        if not candidate["available"]:
            reason_codes[name].append(
                "UNAVAILABLE"
            )
            passes = False

        # Quality
        if (
            candidate["quality"]
            < policy["minQuality"]
        ):
            reason_codes[name].append(
                "QUALITY_FLOOR"
            )
            passes = False

        # Freshness
        if (
            policy["freshnessRequired"]
            and not candidate["freshness"]
        ):
            reason_codes[name].append(
                "FRESHNESS_REQUIRED"
            )
            passes = False

        # Latency
        if (
            candidate["latencyMs"]
            > policy["maxLatencyMs"]
        ):
            reason_codes[name].append(
                "LATENCY_LIMIT"
            )
            passes = False

        # Memory
        if (
            candidate["memoryMb"]
            > policy["maxMemoryMb"]
        ):
            reason_codes[name].append(
                "MEMORY_LIMIT"
            )
            passes = False

        # Labeled data
        if (
            candidate["labeledExamples"]
            > policy["maxLabeledExamples"]
        ):
            reason_codes[name].append(
                "DATA_LIMIT"
            )
            passes = False

        # Cost
        if cost > policy["maxTotalCost"]:
            reason_codes[name].append(
                "COST_LIMIT"
            )
            passes = False

        if passes:
            eligible.append(name)

    # -----------------------------------------------------
    # Final choose result
    # -----------------------------------------------------

    for name in INTERVENTIONS:

        reason_codes[name] = utf8_sort_unique(
            reason_codes[name]
        )

        if name not in total_costs:
            total_costs[name] = 0

    return {
        "selected": (
            eligible[0]
            if eligible
            else None
        ),
        "eligible": eligible,
        "totalCosts": {
            name: total_costs[name]
            for name in INTERVENTIONS
        },
        "reasonCodes": {
            name: reason_codes[name]
            for name in INTERVENTIONS
        },
    }


# =========================================================
# REPAIR OPERATION
# =========================================================

def repair_operation(data):

    reason_codes = []

    # =====================================================
    # TOKENS
    # =====================================================

    tokens = data.get("tokens")

    token_valid = (
        isinstance(tokens, list)
        and len(tokens) > 0
    )

    if token_valid:

        for token in tokens:

            if not isinstance(token, dict):
                token_valid = False
                break

            if not is_safe_int(
                token.get("id")
            ):
                token_valid = False
                break

            if token.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                token_valid = False
                break

            if not isinstance(
                token.get("padding"),
                bool
            ):
                token_valid = False
                break

            if not isinstance(
                token.get("text"),
                str
            ):
                token_valid = False
                break

    if token_valid:

        labels = []

        for token in tokens:

            if (
                token["role"] == "assistant"
                and token["padding"] is False
            ):
                labels.append(
                    token["id"]
                )
            else:
                labels.append(-100)

    else:

        labels = (
            [-100] * len(tokens)
            if isinstance(tokens, list)
            else []
        )

        reason_codes.append(
            "INVALID_TOKEN"
        )

    # =====================================================
    # TEMPLATE
    # =====================================================

    template_pass = (
        data.get("templateApplications") == 1
    )

    if not template_pass:
        reason_codes.append(
            "CHAT_TEMPLATE_COUNT"
        )

    # =====================================================
    # PARAMETERS
    # =====================================================

    parameters = data.get("parameters")
    allowed_targets = data.get("allowedTargets")

    parameter_valid = True

    if not isinstance(parameters, list):
        parameter_valid = False

    if not isinstance(
        allowed_targets,
        list
    ):
        parameter_valid = False

    elif len(allowed_targets) == 0:
        parameter_valid = False

    elif any(
        not isinstance(x, str) or x == ""
        for x in allowed_targets
    ):
        parameter_valid = False

    elif len(set(allowed_targets)) != len(
        allowed_targets
    ):
        parameter_valid = False

    parameter_names = set()
    trainable = []

    if parameter_valid:

        for parameter in parameters:

            if not isinstance(
                parameter,
                dict
            ):
                parameter_valid = False
                break

            name = parameter.get("name")
            target = parameter.get("target")
            numel = parameter.get("numel")

            # Name
            if (
                not isinstance(name, str)
                or name == ""
            ):
                parameter_valid = False
                break

            # Target
            if (
                not isinstance(target, str)
                or target == ""
            ):
                parameter_valid = False
                break

            # Numel
            if not is_positive_safe_int(
                numel
            ):
                parameter_valid = False
                break

            # Unique names
            if name in parameter_names:
                parameter_valid = False
                break

            parameter_names.add(name)

            # -------------------------------------------------
            # LoRA trainability
            # -------------------------------------------------

            is_lora_weight = (
                name.endswith(
                    ".lora_A.weight"
                )
                or name.endswith(
                    ".lora_B.weight"
                )
            )

            if (
                target in allowed_targets
                and is_lora_weight
            ):
                trainable.append(
                    parameter
                )

    if not parameter_valid:
        reason_codes.append(
            "INVALID_PARAMETER"
        )

    # Must have at least one trainable LoRA parameter
    if (
        parameter_valid
        and len(trainable) == 0
    ):
        parameter_valid = False

        reason_codes.append(
            "INVALID_PARAMETER"
        )

    # =====================================================
    # TRAINABLE PARAMETER SORTING
    # =====================================================

    trainable_params = utf8_sort([
        parameter["name"]
        for parameter in trainable
    ])

    # =====================================================
    # SAFE TRAINABLE COUNT
    # =====================================================

    trainable_count = 0

    if parameter_valid:

        for parameter in trainable:

            numel = parameter["numel"]

            if (
                trainable_count
                > SAFE_INT_MAX - numel
            ):
                parameter_valid = False
                break

            trainable_count += numel

    if (
        not parameter_valid
        and "INVALID_PARAMETER"
        not in reason_codes
    ):
        reason_codes.append(
            "INVALID_PARAMETER"
        )

    # =====================================================
    # INFERENCE MODE
    # =====================================================

    inference_mode = data.get(
        "inferenceMode"
    )

    if inference_mode is not False:
        reason_codes.append(
            "INFERENCE_MODE"
        )

    # =====================================================
    # PEFT CONFIG
    # =====================================================

    peft_config_pass = (
        parameter_valid
        and inference_mode is False
    )

    # =====================================================
    # ARTIFACT FILES
    # =====================================================

    artifact_files = data.get(
        "artifactFiles"
    )

    expected_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    adapter_files = []

    if isinstance(
        artifact_files,
        list
    ):

        exact_files = (
            len(artifact_files) == 2
            and all(
                isinstance(x, str)
                for x in artifact_files
            )
            and len(set(artifact_files)) == 2
            and set(artifact_files)
            == expected_files
        )

        if exact_files:

            adapter_files = utf8_sort(
                artifact_files
            )

        else:

            reason_codes.append(
                "ADAPTER_FILE_SET"
            )

    else:

        reason_codes.append(
            "ADAPTER_FILE_SET"
        )

    # =====================================================
    # FULL MODEL ARTIFACT
    # =====================================================

    full_model_files = {
        "pytorch_model.bin",
        "pytorch_model.safetensors",
        "model.bin",
        "model.safetensors",
    }

    if isinstance(
        artifact_files,
        list
    ):

        if any(
            isinstance(x, str)
            and x in full_model_files
            for x in artifact_files
        ):
            reason_codes.append(
                "FULL_MODEL_ARTIFACT"
            )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    checkpoint = data.get(
        "checkpoint"
    )

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
        and required_checkpoint.issubset(
            checkpoint.keys()
        )
    )

    if not checkpoint_complete:
        reason_codes.append(
            "INCOMPLETE_CHECKPOINT"
        )

    # =====================================================
    # LINEAGE
    # =====================================================

    base_revision = data.get(
        "baseRevision"
    )

    dataset_digest = data.get(
        "datasetDigest"
    )

    code_digest = data.get(
        "codeDigest"
    )

    config_digest = data.get(
        "configDigest"
    )

    expected_digests = data.get(
        "expectedDigests"
    )

    lineage_pass = True

    # Base revision
    if (
        not isinstance(
            base_revision,
            str
        )
        or HEX40.fullmatch(
            base_revision
        ) is None
    ):

        reason_codes.append(
            "MUTABLE_BASE_REVISION"
        )

        lineage_pass = False

    # Digests
    for digest in [
        dataset_digest,
        code_digest,
        config_digest,
    ]:

        if (
            not isinstance(
                digest,
                str
            )
            or HEX64.fullmatch(
                digest
            ) is None
        ):
            lineage_pass = False

    # Expected digest matching
    if not isinstance(
        expected_digests,
        dict
    ):

        lineage_pass = False

    else:

        if (
            expected_digests.get(
                "datasetDigest"
            )
            != dataset_digest
        ):
            lineage_pass = False

        if (
            expected_digests.get(
                "codeDigest"
            )
            != code_digest
        ):
            lineage_pass = False

        if (
            expected_digests.get(
                "configDigest"
            )
            != config_digest
        ):
            lineage_pass = False

    if (
        not lineage_pass
        and "MUTABLE_BASE_REVISION"
        not in reason_codes
    ):
        reason_codes.append(
            "LINEAGE_MISMATCH"
        )

    # =====================================================
    # EFFECTIVE BATCH
    # =====================================================

    micro_batch = data.get(
        "microBatch"
    )

    gradient_accumulation = data.get(
        "gradientAccumulation"
    )

    replicas = data.get(
        "replicas"
    )

    expected_batch = data.get(
        "expectedEffectiveBatch"
    )

    batch_pass = (
        is_positive_safe_int(
            micro_batch
        )
        and is_positive_safe_int(
            gradient_accumulation
        )
        and is_positive_safe_int(
            replicas
        )
        and is_positive_safe_int(
            expected_batch
        )
    )

    if batch_pass:

        effective_batch = (
            micro_batch
            * gradient_accumulation
            * replicas
        )

        if (
            effective_batch > SAFE_INT_MAX
            or effective_batch
            != expected_batch
        ):
            batch_pass = False

    if not batch_pass:
        reason_codes.append(
            "EFFECTIVE_BATCH_MISMATCH"
        )

    # =====================================================
    # EVALUATION ISOLATION
    # =====================================================

    train_ids = data.get(
        "trainRowIds"
    )

    eval_ids = data.get(
        "evalRowIds"
    )

    ids_valid = (
        isinstance(train_ids, list)
        and isinstance(eval_ids, list)

        and len(train_ids) > 0
        and len(eval_ids) > 0

        and all(
            isinstance(x, str)
            and x != ""
            for x in train_ids
        )

        and all(
            isinstance(x, str)
            and x != ""
            for x in eval_ids
        )

        and len(set(train_ids))
        == len(train_ids)

        and len(set(eval_ids))
        == len(eval_ids)
    )

    eval_isolated = (
        ids_valid
        and set(train_ids).isdisjoint(
            set(eval_ids)
        )
    )

    if not eval_isolated:
        reason_codes.append(
            "EVAL_LEAKAGE"
        )

    # =====================================================
    # EVAL DROPOUT
    # =====================================================

    dropout_pass = (
        data.get(
            "dropoutActiveDuringEval"
        )
        is False
    )

    if not dropout_pass:
        reason_codes.append(
            "EVAL_DROPOUT_ACTIVE"
        )

    # =====================================================
    # RESUME
    # =====================================================

    uninterrupted = data.get(
        "uninterruptedWeights"
    )

    resumed = data.get(
        "resumedWeights"
    )

    tolerance = data.get(
        "resumeTolerance"
    )

    resume_pass = (
        isinstance(
            uninterrupted,
            list
        )
        and isinstance(
            resumed,
            list
        )

        and len(uninterrupted) > 0

        and len(uninterrupted)
        == len(resumed)

        and all(
            finite_number(x)
            for x in uninterrupted
        )

        and all(
            finite_number(x)
            for x in resumed
        )

        and finite_number(
            tolerance
        )

        and tolerance >= 0
    )

    if resume_pass:

        for a, b in zip(
            uninterrupted,
            resumed
        ):

            if (
                abs(
                    float(a) - float(b)
                )
                > float(tolerance)
            ):
                resume_pass = False
                break

    if not resume_pass:
        reason_codes.append(
            "RESUME_DIVERGENCE"
        )

    # =====================================================
    # DETERMINISTIC EVALUATION
    # =====================================================

    evaluation_deterministic = (
        dropout_pass
        and resume_pass
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "labels": labels,

        "templatePass": template_pass,

        "trainableParams": trainable_params,

        "trainableCount": trainable_count,

        "peftConfigPass": peft_config_pass,

        "adapterFiles": adapter_files,

        "checkpointComplete":
            checkpoint_complete,

        "lineagePass":
            lineage_pass,

        "evalIsolated":
            eval_isolated,

        "evaluationDeterministic":
            evaluation_deterministic,

        "resumePass":
            resume_pass,

        "reasonCodes":
            utf8_sort_unique(reason_codes),
    }


# =========================================================
# POST /adapt
# =========================================================

@app.post("/adapt")
async def adapt(data: dict[str, Any]):

    operation = data.get(
        "operation"
    )

    # Unknown / missing operation
    if operation not in {
        "choose",
        "repair",
    }:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if operation == "choose":
        return choose_operation(data)

    return repair_operation(data)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok"
    }
