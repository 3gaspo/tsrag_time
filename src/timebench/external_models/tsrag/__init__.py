"""Source-adapted TS-RAG ARM and upstream-faithful retriever.

The ARM computation follows UConn-DSIS/TS-RAG at commit
73ac807789d2e61b8a3dfc8514e3fc947fe185cc. Unrelated ARM training and alternate
augmentation paths are removed, but the released MoE computation is retained.
The retriever preserves the released Chronos-T5 EOS, FAISS IndexFlatL2, and
top-k rule; the experiment changes only which causal dates enter its index.
"""

