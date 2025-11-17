LLaDAModelLM(
  (model): LLaDAModel(
    (transformer): ModuleDict(
      (wte): Embedding(126464, 4096)
      (emb_drop): Dropout(p=0.0, inplace=False)
      (ln_f): RMSLayerNorm()
      (blocks): ModuleList(
        (0-31): 32 x LLaDALlamaBlock(
          (dropout): Dropout(p=0.0, inplace=False)
          (act): SiLU()
          (attn_out): Linear(in_features=4096, out_features=4096, bias=False)
          (ff_out): Linear(in_features=12288, out_features=4096, bias=False)
          (rotary_emb): RotaryEmbedding()
          (attn_norm): RMSLayerNorm()
          (ff_norm): RMSLayerNorm()
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (ff_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
        )
      )
      (ff_out): Linear(in_features=4096, out_features=126464, bias=False)
    )
  )
)
