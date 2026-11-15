"""How to ask each kind of decision model one `choice` question.

Every backend takes the argparse namespace and has
    ask(rec, img) -> ({option id: probability}, input token count)
where rec is a /v1/systemone request (state + one question "route") and img is a PIL image or None.

  systemone  any server with POST /v1/systemone (vLLM / SGLang / llama.cpp / ollama)
  clef       Cloudflare Clef family in this process (official Cloudflare loader code)
  smaug      abacusai/Smaug-Decisions (Gemma 4 E4B): LM-head logits of the letters A..T, max 20 options
  intern     internlm/Intern-Decision-* (Qwen3.5): logits before the <decision> marker, max 62 options
  edgellm    Smaug / Intern-Decision as a TensorRT Edge-LLM engine (--engine-dir)

The smaug / intern prompts are written here from the model cards and the authors' reference scripts
(smaug_decide.py, inference.py); those scripts are not run. Calibration temperatures are left out:
they do not change the answer.

To add a model that no runtime serves: write a class here with __init__(self, a) (load the model; `a.model` is
the repo given with --model) and ask(rec, img) as above, add it to BACKENDS, and list it in tools/acc_lists/
as `host:<key> <name> <precision> <repo>`. Check it first with `--limit 2` (bench/eval_acc.py).
"""
import base64, io, json, math, os, sys, urllib.request


def _criteria(rec):
    q = rec["questions"]["route"]
    return q["instructions"], list(q["criteria"].items())


def _body_text(rec):
    """The free text of the state (eval_v1 / bench2 records keep it in <key>.body)."""
    return " ".join(v["body"] for v in rec["state"].values() if isinstance(v, dict) and "body" in v)


class SystemOne:
    def __init__(self, a):
        self.url, self.model, self.raw = a.url, a.model, a.image_raw

    def ask(self, rec, img):
        body = {"model": self.model, "state": rec["state"], "questions": rec["questions"]}
        if img is not None:
            buf = io.BytesIO(); img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            body["images"] = [b64 if self.raw else "data:image/png;base64," + b64]
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        return out["answers"]["route"]["probabilities"], out.get("usage", {}).get("input_tokens")


class Clef:
    def __init__(self, a):
        import torch
        try:
            import gptqmodel  # noqa: F401  (must be imported before flash-linear-attention, see bench2.py)
        except ImportError:
            pass
        from huggingface_hub import snapshot_download
        code_repo = a.code_repo or a.model
        assert code_repo.startswith("Cloudflare/"), "loader code must come from an official Cloudflare repo"
        sys.path.insert(0, snapshot_download(code_repo))
        from joint_schema_model import collate_records, encode_record, load_release_model
        kw = {}
        if a.bnb:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if a.bnb == "int8" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=torch.bfloat16))
        self.model, self.proc = load_release_model(snapshot_download(a.model), device="cuda", **kw)
        self.torch, self.enc, self.collate = torch, encode_record, collate_records

    def ask(self, rec, img):
        torch = self.torch
        if img is not None:
            rec = {**rec, "images": [img]}
        enc = self.enc(self.proc.tokenizer, rec, processor=self.proc, max_length=64000)
        batch = self.collate([enc], self.proc.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            probs = self.model(batch)[0][0].float().softmax(-1).cpu().tolist()
        torch.cuda.synchronize()
        return dict(zip(enc.questions[0].option_ids, probs)), len(enc.input_ids)


class Smaug:
    LETTERS = "ABCDEFGHIJKLMNOPQRST"
    PREAMBLE = ("You make decisions for apps. Read the rules, the context and the input, then answer the question by "
                "choosing from the lettered options.")
    IMAGE_TOKENS = 560

    def __init__(self, a):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        path = snapshot_download(a.model, allow_patterns=["*.json", "*.jinja", "model*.safetensors"])
        self.model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained(path)
        tok = self.proc.tokenizer
        self.letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in self.LETTERS],
                                       device="cuda")
        self.cap = json.load(open(f"{path}/readout.json")).get("final_logit_softcapping")
        self.torch = torch

    @classmethod
    def prompt(cls, rec, has_img):
        """The whole prompt (the answer letter comes next) and the options."""
        instr, opts = _criteria(rec)
        assert len(opts) <= len(cls.LETTERS), f"Smaug takes at most {len(cls.LETTERS)} options"
        parts = [cls.PREAMBLE]
        text = _body_text(rec)
        if has_img:  # the text becomes the context, the image is the input
            parts += [f"Context: {text}", "Input: <|image|>"]
        else:
            parts.append(f"Input: {text}")
        parts.append("\n".join([f"Question: {instr}"] + [f"{cls.LETTERS[i]}) {lab}" for i, (_, lab) in enumerate(opts)]))
        parts.append("Answer with the letter of the best option.")
        return "<bos><|turn>user\n" + "\n".join(parts) + "<turn|>\n<|turn>model\n", opts

    def ask(self, rec, img):
        torch = self.torch
        prompt, opts = self.prompt(rec, img is not None)
        kw = {"images": [img], "images_kwargs": {"max_soft_tokens": self.IMAGE_TOKENS}} if img is not None else {}
        enc = self.proc(text=[prompt], return_tensors="pt", add_special_tokens=False, **kw).to("cuda")
        with torch.inference_mode():
            h = self.model.model(**enc, use_cache=False).last_hidden_state[0, -1].float()
            z = h @ self.model.lm_head.weight[self.letter_ids[:len(opts)]].float().T
            if self.cap:
                z = torch.tanh(z / self.cap) * self.cap
            p = z.softmax(-1).cpu().tolist()
        torch.cuda.synchronize()
        return {oid: p[i] for i, (oid, _) in enumerate(opts)}, enc["input_ids"].shape[-1]


class Intern:
    SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    SYSTEM = ("You are a careful decision assistant. Use the state and decision schema in the user message to make the "
              "requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed "
              "options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names "
              "and symbols exactly as given. Do not include explanations, Markdown, or extra text.")

    def __init__(self, a):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
        path = snapshot_download(a.model, allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "*.model"])
        self.tok = AutoTokenizer.from_pretrained(path)
        assert "<decision>" in self.tok.get_added_vocab(), "not an Intern-Decision tokenizer"
        self.marker = self.tok.convert_tokens_to_ids("<decision>")
        self.proc = AutoProcessor.from_pretrained(path); self.proc.tokenizer = self.tok
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16,
                                                                     attn_implementation="sdpa").to("cuda").eval()
        self.sym_ids = [self.tok.encode(s, add_special_tokens=False)[0] for s in self.SYMBOLS]
        self.torch = torch

    @classmethod
    def prompt(cls, rec, has_img, tmpl):
        """The chat-formatted text with one <decision> marker per question (in the assistant JSON skeleton),
        and [(question name, options)]. All questions are answered by one forward pass (as in the authors'
        inference.py). tmpl: the tokenizer (or processor) whose chat template is used."""
        fields, schema = [], []
        for name, q in rec["questions"].items():
            opts = list(q["criteria"].items())
            assert len(opts) <= len(cls.SYMBOLS)
            fields.append((name, opts))
            schema += [f"{name}: {q['instructions']}"] + [f"    {cls.SYMBOLS[i]} = {oid}: {lab}"
                                                          for i, (oid, lab) in enumerate(opts)]
        user = ("Return one answer for every field using the supplied answer symbols.\n\n## State\n"
                + json.dumps(rec["state"], ensure_ascii=False, indent=2) + "\n## Decision schema\n" + "\n".join(schema))
        skeleton = json.dumps(dict.fromkeys([n for n, _ in fields], "<decision>"), ensure_ascii=False, indent=4)
        content = [{"type": "image"}, {"type": "text", "text": user}] if has_img else user
        msgs = [{"role": "system", "content": cls.SYSTEM}, {"role": "user", "content": content},
                {"role": "assistant", "content": skeleton}]
        text = tmpl.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
                                        add_vision_id=True)
        return text, fields

    def ask_all(self, rec, img):
        """Every question of rec in one pass: ({question name: {option id: probability}}, input token count)."""
        torch = self.torch
        text, fields = self.prompt(rec, img is not None, self.proc if img is not None else self.tok)
        enc = (self.proc(text=[text], images=[img], return_tensors="pt") if img is not None
               else self.tok(text, add_special_tokens=False, return_tensors="pt")).to("cuda")
        pos = (enc["input_ids"][0] == self.marker).nonzero().flatten() - 1
        assert len(pos) == len(fields)
        with torch.inference_mode():
            logits = self.model(**enc, use_cache=False, logits_to_keep=pos).logits[0]
            out = {}
            for j, (name, opts) in enumerate(fields):
                p = logits[j, self.sym_ids[:len(opts)]].float().softmax(-1).cpu().tolist()
                out[name] = {oid: p[i] for i, (oid, _) in enumerate(opts)}
        torch.cuda.synchronize()
        return out, enc["input_ids"].shape[-1]

    def ask(self, rec, img):
        out, ntok = self.ask_all(rec, img)
        return out["route"], ntok


class EdgeLLM:
    """A TensorRT Edge-LLM engine (runtimes/edgellm) of Smaug-Decisions (gemma4) or Intern-Decision (qwen3_5).

    The prompt is the same text as the smaug / intern backends, sent without a chat template. The engine generates
    one token; the option symbols get a large equal logit bias so that all of them are among the returned top
    log-probabilities, and their probabilities are renormalized (the bias cancels out).
    """
    BIAS = 100.0

    def __init__(self, a):
        from huggingface_hub import snapshot_download
        from tensorrt_edgellm import runtime
        from transformers import AutoTokenizer
        # --model: an HF repo, or a local checkpoint directory (e.g. one with a patched config.json)
        path = a.model if os.path.isdir(a.model) else snapshot_download(a.model)
        self.kind = json.load(open(f"{path}/config.json"))["model_type"]
        assert self.kind in ("gemma4", "qwen3_5"), f"no decision prompt for model_type {self.kind}"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.symbols = Smaug.LETTERS if self.kind == "gemma4" else Intern.SYMBOLS
        self.sym_ids = [self.tok.encode(s, add_special_tokens=False)[0] for s in self.symbols]
        self.rt = runtime.load()
        vis = os.path.join(a.engine_dir, "visual")
        self.llm = self.rt.LLMRuntime(engine_dir=a.engine_dir, multimodal_engine_dir=vis if os.path.isdir(vis) else "",
                                      checkpoint_dir=path)

    def ask(self, rec, img):
        rt = self.rt
        if self.kind == "gemma4":
            text, opts = Smaug.prompt(rec, img is not None)
        else:
            text, fields = Intern.prompt(rec, img is not None, self.tok)
            assert len(fields) == 1, "the edgellm backend reads one question per request"
            opts = fields[0][1]
            text = text[:text.index("<decision>")]
        ids = self.sym_ids[:len(opts)]
        req = rt.create_generation_request([[rt.create_text_message("user", text)]], temperature=0.0, top_k=1,
                                           max_generate_length=1, apply_chat_template=False,
                                           logit_bias={i: self.BIAS for i in ids}, num_logprobs=min(50, len(opts)))
        if img is not None:
            buf = io.BytesIO(); img.save(buf, format="PNG")
            req.requests[0].image_buffers = [rt.load_image_from_bytes(buf.getvalue())]
        res = self.llm.handle_request(req)
        lp = {e.token_id: e.logprob for e in res.logprobs[0][0]}
        z = [math.exp(lp[i]) if i in lp else 0.0 for i in ids]
        s = sum(z)
        return {oid: z[k] / s for k, (oid, _) in enumerate(opts)}, res.prompt_token_counts[0]


BACKENDS = {"systemone": SystemOne, "clef": Clef, "smaug": Smaug, "intern": Intern, "edgellm": EdgeLLM}
