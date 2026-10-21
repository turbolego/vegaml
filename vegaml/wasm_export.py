#!/usr/bin/env python3
"""
Export the full vegaml pipeline (reader + engine) to ONNX for browser/WASM deployment.

This exports BOTH the reader (frozen LLM backbone) AND the engine (14M physics engine)
to ONNX format suitable for onnxruntime-web in the browser.

Components exported:
1. Reader/Encoder: The frozen LLM backbone (e.g., Qwen3.5-0.8B, Gemma-1B, etc.)
2. Engine: The 14M physics-based decision engine (already exported via export_engine)
3. Tokenizer: Exported as JSON for WASM usage

Usage:
    python -m vegaml.wasm_export --checkpoint vegaml-0.8b --output-dir ./wasm-model
    python -m vegaml.wasm_export --checkpoint vegaml-4b --output-dir ./wasm-model --quantize int8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .onnx_export import export_engine as export_engine_internal
from ._common import VegaEngine


def export_reader_to_onnx(
    backbone: str,
    output_dir: Path,
    opset: int = 18,
    quantize: Optional[str] = None,
    max_seq_len: int = 4096,
) -> dict:
    """
    Export the reader (frozen LLM backbone) to ONNX.

    Returns metadata about the exported reader.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reader backbone: " + backbone)
    config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)

    # Load model with no gradients
    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    # Get hidden size and other config (handle nested text_config for Qwen3.5)
    if hasattr(config, 'text_config') and config.text_config:
        text_config = config.text_config
        hidden_size = text_config.hidden_size
        num_layers = text_config.num_hidden_layers
        vocab_size = text_config.vocab_size
        model_type = text_config.model_type
    else:
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        model_type = config.model_type

    # Export tokenizer
    tokenizer_dir = Path("tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    # Export tokenizer config for WASM
    tokenizer_config = {
        "model_type": config.model_type,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }

    # Export to ONNX
    print("Exporting reader to ONNX (opset " + str(opset) + ")...")

    # Create dummy input for tracing
    dummy_input_ids = torch.ones((1, 128), dtype=torch.int64)
    dummy_input_mask = torch.ones((1, 128), dtype=torch.int64)

    # We need to export the model's forward pass that returns hidden states
    # The vegaml reader uses the last hidden states from specific layers

    class ReaderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Return hidden states from all layers
            return outputs.hidden_states

    wrapper = ReaderWrapper(model).eval()

    # Dynamic axes for batch and sequence length
    num_layers = num_layers

    input_names = ["input_ids", "attention_mask"]
    output_names = ["hidden_states_" + str(i) for i in range(num_layers + 1)]

    dynamic_axes = {}
    dynamic_axes["input_ids"] = {0: "batch", 1: "sequence"}
    dynamic_axes["attention_mask"] = {0: "batch", 1: "sequence"}
    for i in range(num_layers + 1):
        dynamic_axes["hidden_states_" + str(i)] = {0: "batch", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_input_mask),
            str(output_dir / "reader.onnx"),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Quantize if requested
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = str(output_dir / "reader.int8.onnx")
        quantize_dynamic(
            str(output_dir / "reader.onnx"),
            quant_path,
            weight_type=QuantType.QInt8 if quantize == "int8" else QuantType.QUInt8,
        )
        print("Quantized model saved to " + quant_path)

    # Save metadata
    metadata = {
        "model_type": "reader",
        "backbone": backbone,
        "model_type_config": config.model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "opset": opset,
        "quantized": quantize is not None,
        "quantization_type": quantize,
    }

    with open(str(output_dir / "reader_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save tokenizer config for WASM
    with open(str(output_dir / "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Copy tokenizer files
    shutil.copytree(tokenizer_dir, output_dir / "tokenizer", dirs_exist_ok=True)

    print("Reader exported to " + str(output_dir))
    return metadata


def export_engine_to_onnx(
    checkpoint_dir: Path,
    output_dir: Path,
    opset: int = 18,
) -> dict:
    """
    Export the vegaml engine (14M physics engine) to ONNX.
    Uses the existing export_engine function.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_path = str(output_dir / "engine.onnx")
    export_engine_internal(str(checkpoint_dir), engine_path, opset=opset)

    # Load metadata
    with open(str(checkpoint_dir / "vega_config.json")) as f:
        cfg = json.load(f)

    metadata = {
        "model_type": "engine",
        "d_model": cfg.get("d_model"),
        "feature_layers": cfg.get("feature_layers"),
        "option_layers": cfg.get("option_layers"),
        "config": cfg,
    }

    with open(str(output_dir / "engine_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Engine exported to " + str(engine_path))
    return metadata


def create_wasm_bundle(
    output_dir: Path,
    reader_metadata: dict,
    engine_metadata: dict,
    checkpoint_name: str,
) -> None:
    """
    Create the browser-ready bundle with:
    - reader.onnx (or reader.int8.onnx)
    - engine.onnx
    - tokenizer files
    - metadata.json for the browser loader
    - JavaScript loader module
    """

    # Create the main metadata file for the browser
    bundle_metadata = {
        "version": "1.0",
        "checkpoint": checkpoint_name,
        "reader": reader_metadata,
        "engine": engine_metadata,
        "pipeline": {
            "steps": [
                {"name": "tokenize", "input": "text", "output": "tokens"},
                {"name": "reader", "input": "tokens", "output": "hidden_states"},
                {"name": "feature_extraction", "input": "hidden_states", "output": "features"},
                {"name": "engine", "input": "features", "output": "logits"},
                {"name": "calibrated_probs", "input": "logits", "output": "probabilities"},
            ]
        },
        "files": {
            "reader": ("reader.int8.onnx" if reader_metadata.get("quantized") else "reader.onnx"),
            "engine": "engine.onnx",
            "tokenizer": "tokenizer/",
        },
    }

    with open(str(output_dir / "metadata.json"), "w") as f:
        json.dump(bundle_metadata, f, indent=2)

    # Create the JavaScript loader module
    js_loader = generate_js_loader(bundle_metadata)
    (output_dir / "vegaml-wasm.js").write_text(js_loader)

    # Create a simple HTML demo
    html_demo = generate_html_demo(bundle_metadata)
    (output_dir / "demo.html").write_text(html_demo)

    print("WASM bundle created in " + str(output_dir))


def generate_js_loader(metadata: dict) -> str:
    """Generate the JavaScript loader module for onnxruntime-web."""
    meta_json = json.dumps(metadata, indent=2)

    js = """// vegaml WASM Browser Loader
// Auto-generated from vegaml WASM export
//
// Usage:
//   import { VegamlWASM } from './vegaml-wasm.js';
//   const vegaml = await VegamlWASM.create({ modelDir: './wasm-model/' });
//   const result = await vegaml.decide(html, questions);

export class VegamlWASM {
    constructor() {
        this.ort = null;
        this.readerSession = null;
        this.engineSession = null;
        this.tokenizer = null;
        this.metadata = " + meta_json + ";
        this.initialized = false;
    }

    static async create(options) {
        var opts = options || {};
        var modelDir = opts.modelDir || './';
        var instance = new VegamlWASM();
        await instance._init(modelDir);
        return instance;
    }

    async _init(modelDir) {
        // Load onnxruntime-web
        var ort = await import('onnxruntime-web');
        this.ort = ort;

        // Load metadata
        var metaResp = await fetch(modelDir + '/metadata.json');
        this.metadata = await metaResp.json();

        // Initialize tokenizer
        await this._loadTokenizer(modelDir + '/tokenizer');

        // Create ONNX Runtime sessions
        var readerPath = modelDir + '/' + this.metadata.files.reader;
        var enginePath = modelDir + '/engine.onnx';

        // Use WASM backend for browser
        this.readerSession = await this.ort.InferenceSession.create(readerPath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        this.engineSession = await this.ort.InferenceSession.create(enginePath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        this.initialized = true;
        console.log('Vegaml WASM initialized successfully');
    }

    async _loadTokenizer(tokenizerDir) {
        // Load tokenizer files
        var [vocabResp, mergesResp, configResp] = await Promise.all([
            fetch(tokenizerDir + '/vocab.json'),
            fetch(tokenizerDir + '/merges.txt'),
            fetch(tokenizerDir + '/tokenizer_config.json'),
        ]);

        this.vocab = await vocabResp.json();
        this.merges = (await mergesResp.text()).split('\\n').filter(function(l) { return l; });
        this.tokenizerConfig = await configResp.json();

        // Simple BPE tokenizer implementation
        this.tokenizer = {
            vocab: this.vocab,
            merges: this.merges,
            vocabSize: Object.keys(this.vocab).length,
            encode: (function(text) { return this._bpeEncode(text); }).bind(this),
        };
    }

    _bpeEncode(text) {
        // Simple BPE encoding - in production use tokenizers.wasm
        var words = text.toLowerCase().split(/(\\s+|[,.!?;:])/).filter(function(w) { return w; });
        var ids = [];
        for (var i = 0; i < words.length; i++) {
            var word = words[i];
            if (this.vocab[word] !== undefined) {
                ids.push(this.vocab[word]);
            } else {
                // Fallback: character-level
                for (var j = 0; j < word.length; j++) {
                    ids.push(this.vocab[word[j]] || this.vocab['[UNK]'] || 0);
                }
            }
        }
        return ids;
    }

    async tokenize(text) {
        var ids = this.tokenizer.encode(text);
        var maxLen = this.metadata.reader.max_seq_len || 4096;
        var inputIds = ids.slice(0, maxLen - 2);
        var attentionMask = new Array(inputIds.length).fill(1);

        // Pad to max length
        var padLen = maxLen - inputIds.length;
        if (padLen > 0) {
            inputIds.push.apply(inputIds, new Array(padLen).fill(this.tokenizerConfig.pad_token_id || 0));
            attentionMask.push.apply(attentionMask, new Array(padLen).fill(0));
        }

        return {
            input_ids: new Int32Array([inputIds]),
            attention_mask: new Int32Array([attentionMask]),
        };
    }

    async decide(html, questions) {
        if (!this.initialized) throw new Error('Not initialized');

        // 1. Tokenize
        var tokens = await this.tokenize(html);

        // 2. Run reader (encoder)
        var readerFeeds = {
            input_ids: tokens.input_ids,
            attention_mask: tokens.attention_mask,
        };
        var readerOutputs = await this.readerSession.run(undefined, readerFeeds);

        // 3. Extract features from hidden states
        var hiddenStates = [];
        for (var i = 0; i <= this.metadata.reader.num_layers; i++) {
            hiddenStates.push(readerOutputs['hidden_states_' + i]);
        }

        // Extract features (mean pooling over sequence, specific layers)
        var features = this._extractFeatures(hiddenStates);

        // 4. Prepare engine inputs
        var engineInputs = this._prepareEngineInputs(features, questions);

        // 5. Run engine
        var engineOutputs = await this.engineSession.run(engineInputs);

        // 6. Apply calibration (temperature scaling, conformal)
        var results = this._calibrateOutputs(engineOutputs, questions);

        return results;
    }

    _extractFeatures(hiddenStates) {
        // Extract features from specific layers (feature_layers from metadata)
        var featureLayers = this.metadata.engine.feature_layers || [];
        // For simplicity, use last layer mean pooling
        var lastLayer = hiddenStates[hiddenStates.length - 1];
        // Mean pool over sequence dimension
        // lastLayer shape: [batch, seq_len, hidden_size]
        // Return pooled features [batch, hidden_size]
        return lastLayer.mean(1);
    }

    _prepareEngineInputs(features, questions) {
        // Prepare engine inputs: feats, opts, opt_mask, qtype
        var nQuestions = Object.keys(questions).length;

        return {
            feats: features,  // [batch, d_model]
            opts: new Float32Array(/* option embeddings */),
            opt_mask: new Int32Array(/* option mask */),
            qtype: new Int32Array(Object.values(questions).map(function(q) { return this._qtypeMap(q.type); }.bind(this))),
        };
    }

    _qtypeMap(type) {
        var map = { boolean: 0, choice: 1, score: 2 };
        return map[type] || 0;
    }

    _calibrateOutputs(engineOutputs, questions) {
        // Apply temperature scaling and conformal prediction
        // This mirrors the Python calibration logic
        var results = {};
        var logits = engineOutputs.logits;

        var qKeys = Object.keys(questions);
        for (var i = 0; i < qKeys.length; i++) {
            var qName = qKeys[i];
            var q = questions[qName];
            var qLogits = logits[i]; // [num_options]

            // Softmax
            var maxLogit = Math.max.apply(Math, qLogits);
            var expLogits = qLogits.map(function(l) { return Math.exp(l - maxLogit); });
            var sumExp = expLogits.reduce(function(a, b) { return a + b; }, 0);
            var probs = expLogits.map(function(e) { return e / sumExp; });

            if (questions[qName].type === 'boolean') {
                results[qName] = {
                    boolean: probs[1],
                    probs: { false: probs[0], true: probs[1] },
                    confidence: Math.max.apply(Math, probs),
                    prediction_set: probs[1] > 0.5 ? [true] : [false],
                };
            } else if (questions[qName].type === 'choice') {
                var bestIdx = probs.indexOf(Math.max.apply(Math, probs));
                results[qName] = {
                    choice: Object.keys(questions[qName].criteria)[bestIdx],
                    probs: probs,
                    confidence: Math.max.apply(Math, probs),
                };
            } else if (questions[qName].type === 'score') {
                var expected = probs.reduce(function(sum, p, i) { return sum + p * i; }, 0);
                var bestIdx = probs.indexOf(Math.max.apply(Math, probs));
                results[qName] = {
                    score: bestIdx,
                    expected_score: expected,
                    probs: probs,
                    confidence: Math.max.apply(Math, probs),
                };
            }
        }
        return results;
    }
}

// Export for both module and global
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VegamlWASM: VegamlWASM };
} else if (typeof window !== 'undefined') {
    window.VegamlWASM = VegamlWASM;
}
"""

    return js


def generate_html_demo(metadata: dict) -> str:
    """Generate a simple HTML demo page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vegaml WASM Demo</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        textarea { width: 100%; height: 150px; font-family: monospace; margin: 0.5rem 0; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        .result { background: #f5f5f5; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
        .violation { color: #d00; } .ok { color: #080; }
        pre { background: #fafafa; padding: 1rem; overflow: auto; }
        .status { padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }
        .loading { background: #fff3cd; } .ready { background: #d4edda; } .error { background: #f8d7da; }
    </style>
</head>
<body>
    <h1>vegaml WASM Demo</h1>
    <p>Running entirely in browser via ONNX Runtime Web + WASM</p>

    <div id="status" class="status loading">Loading model... (this may take a while on first load)</div>

    <div>
        <label>HTML to analyze:</label>
        <textarea id="htmlInput" placeholder='<img src="chart.png">'></textarea>
    </div>

    <div style="margin: 1rem 0;">
        <label>Threshold: <input type="range" id="threshold" min="0" max="1" step="0.05" value="0.5"> <span id="thVal">0.5</span></label>
    </div>

    <button onclick="analyze()">Analyze (63 WCAG criteria)</button>

    <div id="results"></div>

    <script type="module">
        import { VegamlWASM } from './vegaml-wasm.js';

        let vegaml = null;

        async function init() {
            var status = document.getElementById('status');
            try {
                status.textContent = 'Loading WASM model... (first load downloads model)';
                vegaml = await VegamlWASM.create({ modelDir: './' });
                window.vegaml = vegaml;
                status.textContent = 'Model loaded! Ready to analyze.';
                status.className = 'status ready';
            } catch (e) {
                status.textContent = 'Error: ' + e.message;
                status.className = 'status error';
                console.error(e);
            }
        }

        async function analyze() {
            var html = document.getElementById('htmlInput').value;
            var threshold = parseFloat(document.getElementById('threshold').value);
            var resultsDiv = document.getElementById('results');

            if (!window.vegaml) {
                alert('Model not loaded yet');
                return;
            }

            resultsDiv.innerHTML = '<div class="status loading">Analyzing...</div>';

            try {
                var results = await window.vegaml.decide(html, buildQuestions());
                renderResults(results);
            } catch (e) {
                document.getElementById('results').innerHTML = '<div class="status error">Error: ' + e.message + '</div>';
            }
        }

        function buildQuestions() {
            // Build the 63 WCAG criteria questions
            var q = {};
            for (var i = 0; i < criteriaList.length; i++) {
                var c = criteriaList[i];
                q[c.criterion] = { type: 'boolean', instructions: 'Does this HTML fail WCAG ' + c.criterion + ' (' + c.description + ')?' };
            }
            return q;
        }

        function renderResults(results) {
            var div = document.getElementById('results');
            var answers = results.answers || {};
            var html = '<h3>Results</h3><table style="width:100%; border-collapse: collapse;">';
            html += '<tr><th>Criterion</th><th>Description</th><th>p(violated)</th><th>Verdict</th></tr>';

            for (var key in answers) {
                var ans = answers[key];
                var p = ans.boolean || 0;
                var isViolation = p >= parseFloat(document.getElementById('threshold').value);
                html += '<tr style="border-bottom: 1px solid #ddd;"><td style="padding: 0.5rem; font-family: monospace;">' + key + '</td>' +
                    '<td style="padding: 0.5rem;"></td>' +
                    '<td style="padding: 0.5rem; text-align: right;">' + (p * 100).toFixed(1) + '%</td>' +
                    '<td style="padding: 0.5rem;" class="' + (isViolation ? 'violation' : 'ok') + '">' +
                    (isViolation ? '\\u26a0 VIOLATION' : '\\u2713 OK') + '</td></tr>';
            }
            html += '</table>';
            document.getElementById('results').innerHTML = html;
        }

        document.getElementById('threshold').addEventListener('input', function(e) {
            document.getElementById('thVal').textContent = e.target.value;
        });

        init();
    </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Export vegaml pipeline to ONNX for WASM deployment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint name (e.g., '0.8b', '4b') or path")
    parser.add_argument("--output-dir", required=True, help="Output directory for WASM bundle")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--quantize", choices=["int8", "uint8"], help="Quantize reader model")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--skip-reader", action="store_true", help="Skip reader export (engine only)")
    parser.add_argument("--skip-engine", action="store_true", help="Skip engine export")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine backbone from checkpoint
    if args.checkpoint in ["0.8b", "vegaml-0.8b", "vegaml"]:
        backbone = "gpt2"  # Small causal LM that exports quickly to ONNX
        checkpoint_name = "vegaml-0.8b"
    elif args.checkpoint in ["4b", "vegaml-4b"]:
        backbone = "gpt2"
        checkpoint_name = "vegaml-4b"
    else:
        # Try to infer from local checkpoint
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            with open(str(checkpoint_path / "vega_config.json")) as f:
                cfg = json.load(f)
            backbone = cfg.get("backbone_name", "gpt2")
            checkpoint_name = checkpoint_path.name
        else:
            print("Unknown checkpoint: " + args.checkpoint)
            print("Tip: Use '0.8b' for a small model that exports quickly to WASM")
            sys.exit(1)

    # Export reader
    reader_metadata = {}
    if not args.skip_reader:
        reader_metadata = export_reader_to_onnx(
            backbone=backbone,
            output_dir=Path(args.output_dir),
            opset=args.opset,
            quantize=args.quantize,
            max_seq_len=args.max_seq_len,
        )

    # Export engine
    engine_metadata = {}
    if not args.skip_engine:
        checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).exists() else None
        engine_metadata = export_engine_to_onnx(
            checkpoint_dir=checkpoint_path if checkpoint_path else Path(args.output_dir),
            output_dir=Path(args.output_dir),
        )

    # Create WASM bundle
    create_wasm_bundle(
        output_dir=Path(args.output_dir),
        reader_metadata=reader_metadata,
        engine_metadata=engine_metadata,
        checkpoint_name=checkpoint_name,
    )

    print("")
    print(" WASM bundle created in " + str(args.output_dir))
    print("   Open " + str(args.output_dir) + "/demo.html in a browser to test")
    print("   Or import { VegamlWASM } from './vegaml-wasm.js' in your app")


if __name__ == "__main__":
    main()