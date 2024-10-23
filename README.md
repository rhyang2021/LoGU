<div align="center">

# LoGU: Long-form Generation with Uncertainty Expressions

<div>
  <a href='https://hillzhang1999.github.io/' target='_blank'><b>Ruihan Yang</b></a><sup>1</sup>&emsp;
  <a href='https://nealcly.github.io/' target='_blank'>Caiqi Zhang</b></a><sup>2</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=aSJcgQMAAAAJ&hl=en/' target='_blank'>Zhisong Zhang</b></a><sup>2</sup>&emsp;
</div>
<div><sup>1</sup>Fudan University, Shanghai, China</div>
<div><sup>1</sup>Cambridge University</div>
<div><sup>2</sup>Tencent AI Lab</div>

<div>
<h4>

![](https://img.shields.io/badge/PRs-welcome-brightgreen) 
<img src="https://img.shields.io/badge/Version-1.0-blue.svg" alt="Version">
<img src="https://img.shields.io/github/stars/rhyang2021/LoGU?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/rhyang2021/LoGU?color=red" alt="Issues">

</h4>
</div>

<img width="200" alt="image" src="./figures/head.pdf">

</div>

## Introduction

While Large Language Models (LLMs) demonstrate impressive capabilities, they still struggle with generating factually incorrect content (i.e., hallucinations). A promising approach to mitigate this issue is enabling models to express uncertainty when unsure. Previous research on uncertainty modeling has primarily focused on short-form QA, but realworld applications often require much longer responses. In this work, we introduce the task of Long-form Generation with Uncertainty(LoGU). We identify two key challenges: Uncertainty Suppression, where models hesitate to express uncertainty, and Uncertainty Misalignment, where models convey uncertainty inaccurately. To tackle these challenges, we propose a refinement-based data collection framework and a two-stage training pipeline. Our framework adopts a divide-and-conquer strategy, refining uncertainty based on atomic claims. The collected data are then used in training through supervised fine-tuning (SFT) and direct preference optimization (DPO) to enhance uncertainty expression. Extensive experiments on three long-form instruction following datasets show that our method significantly improves accuracy, reduces hallucinations, and maintains the comprehensiveness of responses.

<img width="200" alt="image" src="./figures/main.pdf">

If you are interested in our work, please cite:
```bib
@article{zhang-etal-2023-ICD,
  title     = {LoGU: Long-form Generation with Uncertainty Expressions},
  author    = {Zhang, Yue  and
               Cui, Leyang  and
               Wei, Bi and
               Shuming Shi},
  journal   = {arXiv preprint arXiv:2410.14309},
}
```

## How to Install

You can use the following commands to install the environment for ICD:

```sh
conda create -n icd python==3.10
conda activate icd
pip install -r requirements.txt
cd ./transformers
pip install --editable ./
```

## Run

Try the following command to test our method on TruthfulQA:
```sh
cd ./exp_scripts/benchmark
sh truthfulqa.sh
```

For experiments on Factscore, please try:
```sh
cd ./exp_scripts/benchmark
sh factscore.sh
```
For evaluation on Factscore, please kindly refer to their [repo](https://github.com/shmsw25/FActScore/tree/main).

We also provide some hallucinated models on the huggingface model hub for fast trial:
| Model | Link |
| :------- | :---------: |
| **HillZhang/untruthful_llama2_7b** | [HuggingFace](https://huggingface.co/HillZhang/untruthful_llama2_7b)|
| **HillZhang/untruthful_baichuan2_7b** | [HuggingFace](https://huggingface.co/HillZhang/untruthful_baichuan2_7b)|
| **HillZhang/untruthful_mistral_7b** | [HuggingFace](https://huggingface.co/HillZhang/untruthful_mistral_7b) |
| **HillZhang/untruthful_llama2_7b_bio** | [HuggingFace](https://huggingface.co/HillZhang/untruthful_llama2_7b_bio) |

## Contact

If you have any questions, please feel free to [email](mailto:hillzhang1999@qq.com) me or drop me an issue.
