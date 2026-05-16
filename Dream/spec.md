Dream-Base 模型下的 generation_utils.py 和 modeling_dream.py 目前支持以下几种模式：
0. baseline (use_cache=false, dual_cache=false, focus_decode=false): naive 实现的 dream
1. fast_dllm_prefix_cache (use_cache=true, dual_cache=false, focus_decode=false): 只启用 prompt caching
2. fast_dllm_dual_cache (use_cache=true, dual_cache=true, focus_decode=false): 启用prompt caching和suffix mask block caching
3. focus_dual_cache (use_cache=true, dual_cache=true, focus_decode=true): 启用dual_cache基础上，引入recent topk kv update + attention score topk unmask 机制，减少计算

性能测试：希望扫描以下参数测试四种模式的不同推理性能
1. gen_length: 512, 1024, 2048, 4096，对于 steps，目前设置与 gen_length 相同
2. block_length: 32, 64, 128
3. 对于 focus_decode 选项启用的类型，固定focus_topk=8, focus_layer=3
