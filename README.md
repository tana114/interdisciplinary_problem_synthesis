



### Step1. Synthetic problem generation

Synthesise new, complex problems by providing problems from different fields as seeds. 
In this example, 1,000 synthetic problems were generated using mathematical and physics problems as seeds. 

- Generate drafts for cross-domain problems by providing problems from different fields as seeds.
- Apply methods such as evolvinstruction and B to refine the draft.
- Introduce quantitative indicators to assess the quality of problems, and use these as criteria for revision. Problems with low computational load and high intrinsic difficulty are deemed excellent questions. By revising problems while scoring them, higher-quality synthetic problems are generated.


## under construction




| Step | Implementation Details                            | LLM              | Model                   | Script                      | HF subset                                                  |
|------|---------------------------------------------------|------------------|-------------------------|-----------------------------|------------------------------------------------------------|
| 1    | Synthetic problem generation                      | API (OpenRouter) | "deepseek-r1-0528:free" | problem_gen_manager.py      | OB_PHYS_problem                                            |
| 2    | Answer generation with rollout                    | API (OpenRouter) | "deepseek-r1-0528:free" | rollout_manager.py          | OB_PHYS_rollout                                            |
| 3    | Converting LaTeX-formatted answers to SymPy format | Local (vllm)     | "Qwen3-8b-AWQ"          | sympy_conversion_manager.py | OB_PHYS_rollout_sympy                                      |
| 4    | Verification of self-consistency                  | ----             | ----                    | sympy_self_consistency_manager.py | OB_PHYS_self_consistency, OB_PHYS_self_consistency_rollout |