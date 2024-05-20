# ProbDiff
Code of paper "Language Models can Evaluate Themselves via Probability Discrepancy"


## Overview
We introduce ProbDiff, a self-evaluation technique applicable to any LLM across tasks.
Given a query $q$, ProbDiff first prompts a candidate LLM to generate a response $x$, then asks the LLM to revise $x$ based on $q$, producing a refined response $\hat{x}$. Finally, ProbDiff quantifies the probability discrepancy ${\rm log}\;p({\hat x}|q)$ - ${\rm log}\; p(x|q)$ as the evaluation metric. 
When comparing two candidate LLMs on $q$, a larger probability discrepancy indicates a lower proficiency in handling the instruction.

## Environment
Framework Versions:
```
python=3.10
torch=2.1.2
transformers=4.37.2
vllm=0.3.0
```
## Data
Xiaohongshu Blog writting dataset is stored in `generation_task/data/redbook`

## Scripts
We provide `xxx.sh` to reproduce the results of ProbDiff in each folder. 

## Citation
If you finding our work interesting or helpful to you, please cite this repo.
```
@article{xia2024language,
      title={Language Models can Evaluate Themselves via Probability Discrepancy}, 
      author={Tingyu Xia and Bowen Yu and Yuan Wu and Yi Chang and Chang Zhou},
      journal={arXiv preprint arXiv:2405.10516},
      year={2024}
}
```
