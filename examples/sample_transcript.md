# Sample session transcript

Illustrative example of the agent's step-by-step output format (steps are shown
as `[step N]` lines). Set `OPENAI_API_KEY` and run `python agent.py "..."` to
produce a live session like this.

$ python agent.py "What is 15% of 240, and according to my notes what vector database do we use?"

[step 1] calculator(0.15 * 240)
           -> 36.0
[step 2] file_qa(examples/notes.txt | vector database)
           -> The retrieval layer stores document embeddings in a FAISS vector database.
              ---
              We chunk source documents to ~512 tokens with 64 tokens of overlap, embed
              them with a small open-source embedding model, and index them in FAISS for
              fast approximate nearest-neighbor search.
[step 3] final(...)

Answer: 15% of 240 is 36. According to your notes, the project uses FAISS as its
vector database for storing document embeddings. (Tools used: calculator, file_qa)

And the eval harness (no API key needed — stub mode):

$ python eval.py
mini-agent eval — 5 tasks (backend: stub (no API key))

[PASS] arithmetic       0.00s  1 tool call(s)  ok
[PASS] web_fact         0.00s  1 tool call(s)  ok
[PASS] file_qa          0.00s  1 tool call(s)  ok
[PASS] multi_hop        0.00s  1 tool call(s)  ok
[PASS] no_tool_needed   0.00s  0 tool call(s)  ok

5/5 passed, total latency 0.01s
