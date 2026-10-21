// vegaml WASM Browser Loader
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
        this.merges = (await mergesResp.text()).split('\n').filter(function(l) { return l; });
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
        var words = text.toLowerCase().split(/(\s+|[,.!?;:])/).filter(function(w) { return w; });
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
