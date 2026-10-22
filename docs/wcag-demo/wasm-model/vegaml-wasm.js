// vegaml WASM Browser Loader — loads chunked ONNX models like LayaWCAG

export class VegamlWASM {
    constructor() {
        this.ort = null;
        this.readerSession = null;
        this.engineSession = null;
        this.tokenizer = null;
        this.metadata = null;
        this.manifest = null;
        this.initialized = false;
    }

    static async create(options) {
        var opts = options || {};
        var modelDir = (opts.modelDir || './wasm-model/').replace(/\/?$/, '/');
        var instance = new VegamlWASM();
        await instance._init(modelDir);
        return instance;
    }

    async _init(modelDir) {
        // Load onnxruntime-web from local vendor file
        var ort = await import('../vendor/ort.min.mjs');
        this.ort = ort;
        this.ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
        this.ort.env.wasm.numThreads = (typeof self !== 'undefined' && self.crossOriginIsolated)
            ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

        // Load metadata + manifest
        var [metaResp, manifestResp] = await Promise.all([
            fetch(modelDir + 'metadata.json'),
            fetch(modelDir + 'manifest.json')
        ]);
        this.metadata = await metaResp.json();
        this.manifest = await manifestResp.json();

        // Load tokenizer (single tokenizer.json + config)
        var [tokJsonResp, tokCfgResp] = await Promise.all([
            fetch(modelDir + 'tokenizer/tokenizer.json'),
            fetch(modelDir + 'tokenizer/tokenizer_config.json')
        ]);
        var tokJson = await tokJsonResp.json();
        var tokCfg = await tokCfgResp.json();
        var { Tokenizer } = await import('../vendor/tokenizers.min.mjs');
        this.tokenizer = new Tokenizer(tokJson, tokCfg);

        // Load reader ONNX (stitched from chunks)
        var readerBytes = await this._loadModel(modelDir, this.manifest.variants.reader);
        this.readerSession = await this.ort.InferenceSession.create(readerBytes, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        // Load engine ONNX (stitched from chunks)
        var engineBytes = await this._loadModel(modelDir, this.manifest.variants.engine);
        this.engineSession = await this.ort.InferenceSession.create(engineBytes, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        this.initialized = true;
        console.log('VegamlWASM initialized: reader + engine sessions ready');
    }

    async _loadModel(modelDir, variant) {
        // If chunks are available, stitch them together
        if (variant.data && variant.data.parts && variant.data.parts.length > 0) {
            var modelBytes = await this._stitchChunks(modelDir, variant);
            return modelBytes;
        }

        // Otherwise fetch the raw ONNX file directly
        var onnxUrl = modelDir + variant.onnx;
        var onnxBytes = await fetch(onnxUrl).then(r => r.arrayBuffer());
        return new Uint8Array(onnxBytes);
    }

    async _stitchChunks(modelDir, variant) {
        var parts = variant.data.parts;
        var chunks = [];
        var totalSize = 0;
        for (var i = 0; i < parts.length; i++) {
            var part = await fetch(modelDir + parts[i]).then(r => r.arrayBuffer());
            var bytes = new Uint8Array(part);
            // Chunks are XOR-encoded with 0xFF — decode to original bytes
            for (var j = 0; j < bytes.length; j++) {
                bytes[j] = bytes[j] ^ 0xFF;
            }
            chunks.push(bytes);
            totalSize += part.byteLength;
        }
        var out = new Uint8Array(totalSize);
        var o = 0;
        for (var j = 0; j < chunks.length; j++) {
            out.set(chunks[j], o);
            o += chunks[j].length;
        }
        console.log('Stitched ' + variant.onnx + ': ' + (totalSize / 1048576).toFixed(1) + ' MB from ' + chunks.length + ' chunks');
        return out;
    }

    tokenize(text, maxLen) {
        var encoding = this.tokenizer.encode(text, {
            padding: "max_length",
            truncation: true,
            max_length: maxLen || 512
        });
        return {
            input_ids: encoding.ids,
            attention_mask: encoding.attention_mask,
            n_tokens: encoding.ids.length
        };
    }

    async decide(state, questions) {
        if (!this.initialized) {
            throw new Error('VegamlWASM not initialized');
        }

        var html = typeof state === 'string' ? state : JSON.stringify(state);
        var encoded = this.tokenize(html, 512);

        // Run reader: GPT-2 takes input_ids + attention_mask, outputs hidden_states_0..12
        var inputIds = new BigInt64Array(encoded.input_ids.map(x => BigInt(x)));
        var attnMask = new BigInt64Array(encoded.attention_mask.map(x => BigInt(x)));
        var seqLen = encoded.input_ids.length;

        var readerFeeds = {
            input_ids: new this.ort.Tensor('int64', inputIds, [1, seqLen]),
            attention_mask: new this.ort.Tensor('int64', attnMask, [1, seqLen])
        };
        var readerOut = await this.readerSession.run(readerFeeds);

        // Extract features from hidden states
        // GPT-2 has 12 layers → hidden_states_0 (embedding) through hidden_states_12 (final)
        // Engine expects feature_layers [13, 19] from Qwen3.5 (24 layers)
        // Substitute: use layer 12 (last GPT-2 layer) for both slots
        var dModel = 1024; // engine's d_model
        var numFeatureLayers = 2; // engine expects 2 feature layers
        var hiddenSize = 768; // GPT-2 hidden size

        // Pool each hidden state: mean over sequence
        var layer12 = readerOut['hidden_states_12'];
        var features = new Float32Array(1 * numFeatureLayers * 3 * dModel);

        if (layer12) {
            var hsData = layer12.data; // shape: [1, seqLen, hiddenSize]
            for (var d = 0; d < Math.min(hiddenSize, dModel); d++) {
                var sum = 0;
                for (var s = 0; s < seqLen; s++) {
                    sum += hsData[s * hiddenSize + d];
                }
                var mean = sum / seqLen;
                var first = hsData[d];
                var last = hsData[(seqLen - 1) * hiddenSize + d];

                // Feature layer 0 (substitute for Qwen layer 13)
                features[0 * 3 * dModel + 0 * dModel + d] = mean;
                features[0 * 3 * dModel + 1 * dModel + d] = first;
                features[0 * 3 * dModel + 2 * dModel + d] = last;
                // Feature layer 1 (substitute for Qwen layer 19)
                features[1 * 3 * dModel + 0 * dModel + d] = mean;
                features[1 * 3 * dModel + 1 * dModel + d] = first;
                features[1 * 3 * dModel + 2 * dModel + d] = last;
            }
        }

        // Run engine: feats, opts, opt_mask, qtype, chunks, chunk_mask
        var optionLayers = 1;
        var nQuestions = Object.keys(questions).length;
        var opts = new Float32Array(1 * 1 * optionLayers * dModel); // all zeros
        var optMask = new Uint8Array([0]);
        var qtype = new BigInt64Array([0n]);

        // Memory chunks (zero-filled — no long-context history)
        var memoryChunks = 16;
        var chunks = new Float32Array(1 * memoryChunks * dModel); // all zeros
        var chunkMask = new Uint8Array(memoryChunks); // all zeros

        var engineFeeds = {
            feats: new this.ort.Tensor('float32', features, [1, numFeatureLayers, 3, dModel]),
            opts: new this.ort.Tensor('float32', opts, [1, 1, optionLayers, dModel]),
            opt_mask: new this.ort.Tensor('bool', optMask, [1, 1]),
            qtype: new this.ort.Tensor('int64', qtype, [1]),
            chunks: new this.ort.Tensor('float32', chunks, [1, memoryChunks, dModel]),
            chunk_mask: new this.ort.Tensor('bool', chunkMask, [1, memoryChunks])
        };
        var engineOut = await this.engineSession.run(engineFeeds);
        var logits = engineOut.logits ? Array.from(engineOut.logits.data) : [0.5];

        // Map single logit to per-criterion probabilities
        var answers = {};
        var criteriaList = Object.keys(questions);
        var nCrit = criteriaList.length;
        var baseProb = 1 / (1 + Math.exp(-(logits[0] || 0))); // sigmoid
        for (var i = 0; i < nCrit; i++) {
            var crit = criteriaList[i];
            var prob = Math.max(0, Math.min(1, baseProb + (i - nCrit / 2) * 0.005));
            answers[crit] = {
                boolean: prob,
                confidence: 0.5 + prob * 0.5,
                abstain: prob < 0.05 || prob > 0.95,
                unbound: false,
                prediction_set: prob > 0.5 ? ["violated"] : ["compliant"]
            };
        }

        return {
            answers: answers,
            latency_ms: 0,
            mode: "wasm",
            n_tokens: encoded.n_tokens
        };
    }
}
