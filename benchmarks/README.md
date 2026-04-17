# 0.5.0 Benchmarks

## KV Quantization Benchmark

Run baseline (unquantized KV cache) vs quantized KV cache benchmark:

```bash
python benchmarks/benchmark_kv_quant.py --model_path NS-LLM-0.5/checkpoint-epoch-10 --device cuda
```

Optional quality check using prompt/completion JSON data:

```bash
python benchmarks/benchmark_kv_quant.py --model_path NS-LLM-0.5/checkpoint-epoch-10 --device cuda --eval_data_file generated_data/all.json --eval_max_samples 64
```

Default output report:

- `benchmarks/benchmark_kv_quant_results.json`
