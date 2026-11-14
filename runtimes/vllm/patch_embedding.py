#!/usr/bin/env python3
"""Own patch for vLLM 0.32.0 (written for this benchmark).

Clef-omni's pooler calls VocabParallelEmbedding.forward on the LM head, which lands on the
quantization method's `embedding()`; in 0.32.0 the base class raises NotImplementedError, so
every request to an auto-round (INT4/INT8) clef-omni fails. The LM head of those checkpoints is
bf16 (outside the quantized scope), so a plain gather of its weight is exact.
"""
import pathlib

import vllm

target = pathlib.Path(vllm.__file__).parent / "model_executor/layers/quantization/base_config.py"
src = target.read_text()
old = '''        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError
'''
new = '''        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        # clef-bench patch: bf16 (unquantized) LM head -> plain gather is exact
        weight = getattr(layer, "weight", None)
        if weight is None or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise NotImplementedError
        input_ = args[0] if args else kwargs["input_"]
        return torch.nn.functional.embedding(input_.long(), weight)
'''
assert src.count(old) == 1, "unexpected base_config.py"
target.write_text(src.replace(old, new))
print("patched", target)

# --- patch 2: option limit for the joint schema (Clef-omni) path -------------------------------
# /v1/systemone caps every question at 26 options because the next-token strategies label
# options A..Z. The joint schema head scores each option's span directly and never uses those
# labels, so lift the cap for JointSchemaStrategy only (the benchmark needs 50 / 100 options).
sd = pathlib.Path(vllm.__file__).parent / "entrypoints/generate/structured_decisions"
qt = sd / "question_types.py"
s = qt.read_text()
old_q = "    limit = min(max_options, len(qtype.label_set))\n"
assert s.count(old_q) == 1
qt.write_text(s.replace(old_q, "    limit = max_options  # clef-bench patch: strategies set their own limit\n"))
st = sd / "strategies.py"
s = st.read_text()
old_s = '''        self.suffix = self._tokens(
            "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
            "JOINT SCHEMA DECISIONS:"
        )

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))
'''
assert s.count(old_s) == 1, "unexpected strategies.py"
st.write_text(s.replace(old_s, old_s.replace("max_options=len(LABELS))", "max_options=1024)  # clef-bench patch")))
print("patched", qt, st)
