# Recent LLM Applications

| Domain      | Title | Summary | Published | Link |
|------------|-------|---------|-----------|------|
| Other | ResearchTown: Simulator of Human Research Community |   Large Language Models (LLMs) have demonstrated remarkable potential in
scientific domains, yet a fundamental question remains unanswered: Can we
simulate human research communities with LLMs? Addressing this question can
deepen our understanding of the processes behind idea brainstorming and inspire
the automatic discovery of novel scientific insights. In this work, we propose
ResearchTown, a multi-agent framework for research community simulation. Within
this framework, the human research community is simplified and modeled as an
agent-data graph, where researchers and papers are represented as agent-type
and data-type nodes, respectively, and connected based on their collaboration
relationships. We also introduce TextGNN, a text-based inference framework that
models various research activities (e.g., paper reading, paper writing, and
review writing) as special forms of a unified message-passing process on the
agent-data graph. To evaluate the quality of the research simulation, we
present ResearchBench, a benchmark that uses a node-masking prediction task for
scalable and objective assessment based on similarity. Our experiments reveal
three key findings: (1) ResearchTown can provide a realistic simulation of
collaborative research activities, including paper writing and review writing;
(2) ResearchTown can maintain robust simulation with multiple researchers and
diverse papers; (3) ResearchTown can generate interdisciplinary research ideas
that potentially inspire novel research directions.
 | 2024-12-23T18:26:53Z | [Link](http://arxiv.org/abs/2412.17767v1) |
| Other | In Case You Missed It: ARC 'Challenge' Is Not That Challenging |   ARC Challenge appears more difficult than ARC Easy for modern LLMs primarily
due to an evaluation setup that prevents direct comparison of answer choices
rather than inherent complexity. Although some researchers have quietly shifted
to a more appropriate scheme over the last year, the implications of this
change have yet to be widely acknowledged. We highlight this overlooked shift,
show how similar evaluation practices falsely imply reasoning deficits in other
benchmarks, and demonstrate that fairer methods dramatically reduce performance
gaps (e.g. on SIQA) and even yield superhuman results (OpenBookQA). In doing
so, we reveal how evaluation shapes perceived difficulty and offer guidelines
to ensure that multiple-choice evaluations accurately reflect actual model
capabilities.
 | 2024-12-23T18:14:36Z | [Link](http://arxiv.org/abs/2412.17758v1) |
| Other | ADC: Enhancing Function Calling Via Adversarial Datasets and Code
  Line-Level Feedback |   Large Language Models (LLMs) have made significant strides in Natural
Language Processing and coding, yet they struggle with robustness and accuracy
in complex function calls. To tackle these challenges, this paper introduces
ADC, an innovative approach that enhances LLMs' ability to follow function
formats and match complex parameters. ADC utilizes a high-quality code
fine-tuning dataset with line-level execution feedback, providing granular
process supervision that fosters strong logical reasoning and adherence to
function formats. It also employs an adversarial dataset generation process to
improve parameter matching. The staged training methodology capitalizes on both
enriched code datasets and refined adversarial datasets, leading to marked
improvements in function calling capabilities on the Berkeley Function-Calling
Leaderboard (BFCL) Benchmark. The innovation of ADC lies in its strategic
combination of process supervision, adversarial refinement, and incremental
learning, setting a new standard for LLM proficiency in complex function
calling.
 | 2024-12-23T18:07:18Z | [Link](http://arxiv.org/abs/2412.17754v1) |
| Other | Deliberation in Latent Space via Differentiable Cache Augmentation |   Techniques enabling large language models (LLMs) to "think more" by
generating and attending to intermediate reasoning steps have shown promise in
solving complex problems. However, the standard approaches generate sequences
of discrete tokens immediately before responding, and so they can incur
significant latency costs and be challenging to optimize. In this work, we
demonstrate that a frozen LLM can be augmented with an offline coprocessor that
operates on the model's key-value (kv) cache. This coprocessor augments the
cache with a set of latent embeddings designed to improve the fidelity of
subsequent decoding. We train this coprocessor using the language modeling loss
from the decoder on standard pretraining data, while keeping the decoder itself
frozen. This approach enables the model to learn, in an end-to-end
differentiable fashion, how to distill additional computation into its
kv-cache. Because the decoder remains unchanged, the coprocessor can operate
offline and asynchronously, and the language model can function normally if the
coprocessor is unavailable or if a given cache is deemed not to require extra
computation. We show experimentally that when a cache is augmented, the decoder
achieves lower perplexity on numerous subsequent tokens. Furthermore, even
without any task-specific training, our experiments demonstrate that cache
augmentation consistently reduces perplexity and improves performance across a
range of reasoning-intensive tasks.
 | 2024-12-23T18:02:25Z | [Link](http://arxiv.org/abs/2412.17747v1) |
| Other | RepoTransBench: A Real-World Benchmark for Repository-Level Code
  Translation |   Repository-level code translation refers to translating an entire code
repository from one programming language to another while preserving the
functionality of the source repository. Many benchmarks have been proposed to
evaluate the performance of such code translators. However, previous benchmarks
mostly provide fine-grained samples, focusing at either code snippet, function,
or file-level code translation. Such benchmarks do not accurately reflect
real-world demands, where entire repositories often need to be translated,
involving longer code length and more complex functionalities. To address this
gap, we propose a new benchmark, named RepoTransBench, which is a real-world
repository-level code translation benchmark with an automatically executable
test suite. We conduct experiments on RepoTransBench to evaluate the
translation performance of 11 advanced LLMs. We find that the Success@1 score
(test success in one attempt) of the best-performing LLM is only 7.33%. To
further explore the potential of LLMs for repository-level code translation, we
provide LLMs with error-related feedback to perform iterative debugging and
observe an average 7.09% improvement on Success@1. However, even with this
improvement, the Success@1 score of the best-performing LLM is only 21%, which
may not meet the need for reliable automatic repository-level code translation.
Finally, we conduct a detailed error analysis and highlight current LLMs'
deficiencies in repository-level code translation, which could provide a
reference for further improvements.
 | 2024-12-23T17:52:10Z | [Link](http://arxiv.org/abs/2412.17744v1) |
| Other | YuLan-Mini: An Open Data-efficient Language Model |   Effective pre-training of large language models (LLMs) has been challenging
due to the immense resource demands and the complexity of the technical
processes involved. This paper presents a detailed technical report on
YuLan-Mini, a highly capable base model with 2.42B parameters that achieves
top-tier performance among models of similar parameter scale. Our pre-training
approach focuses on enhancing training efficacy through three key technical
contributions: an elaborate data pipeline combines data cleaning with data
schedule strategies, a robust optimization method to mitigate training
instability, and an effective annealing approach that incorporates targeted
data selection and long context training. Remarkably, YuLan-Mini, trained on
1.08T tokens, achieves performance comparable to industry-leading models that
require significantly more data. To facilitate reproduction, we release the
full details of the data composition for each training phase. Project details
can be accessed at the following link: https://github.com/RUC-GSAI/YuLan-Mini.
 | 2024-12-23T17:47:53Z | [Link](http://arxiv.org/abs/2412.17743v1) |
| Other | Reasoning to Attend: Try to Understand How <SEG> Token Works |   Current Large Multimodal Models (LMMs) empowered visual grounding typically
rely on $\texttt{<SEG>}$ token as a text prompt to jointly optimize the
vision-language model (e.g., LLaVA) and the downstream task-specified model
(\eg, SAM). However, we observe that little research has looked into how it
works.In this work, we first visualize the similarity maps, which are obtained
by computing the semantic similarity between the $\texttt{<SEG>}$ token and the
image token embeddings derived from the last hidden layer in both the LLaVA
encoder and SAM decoder. Intriguingly, we have found that a striking
consistency holds in terms of activation responses in the similarity map,which
reveals that what $\texttt{<SEG>}$ token contributes to is the semantic
similarity within image-text pairs. Specifically, $\texttt{<SEG>}$ token, a
placeholder expanded in text vocabulary, extensively queries among individual
tokenized image patches to match the semantics of an object from text to the
paired image while the Large Language Models (LLMs) are being fine-tuned. Upon
the above findings, we present READ, which facilitates LMMs' resilient
$\textbf{REA}$soning capability of where to atten$\textbf{D}$ under the
guidance of highly activated points borrowed from similarity maps. Remarkably,
READ features an intuitive design, Similarity as Points module (SasP), which
can be seamlessly applied to $\texttt{<SEG>}$-like paradigms in a plug-and-play
fashion.Also, extensive experiments have been conducted on the ReasonSeg and
RefCOCO(+/g) datasets. To validate whether READ suffers from catastrophic
forgetting of previous skills after fine-tuning, we further assess its
generation ability on an augmented FP-RefCOCO(+/g) dataset. All codes and
models are publicly available at https://github.com/rui-qian/READ.
 | 2024-12-23T17:44:05Z | [Link](http://arxiv.org/abs/2412.17741v1) |
| Other | Chumor 2.0: Towards Benchmarking Chinese Humor Understanding |   Existing humor datasets and evaluations predominantly focus on English,
leaving limited resources for culturally nuanced humor in non-English languages
like Chinese. To address this gap, we construct Chumor, the first Chinese humor
explanation dataset that exceeds the size of existing humor datasets. Chumor is
sourced from Ruo Zhi Ba, a Chinese Reddit-like platform known for sharing
intellectually challenging and culturally specific jokes. We test ten LLMs
through direct and chain-of-thought prompting, revealing that Chumor poses
significant challenges to existing LLMs, with their accuracy slightly above
random and far below human. In addition, our analysis highlights that
human-annotated humor explanations are significantly better than those
generated by GPT-4o and ERNIE-4-turbo. We release Chumor at
https://huggingface.co/datasets/dnaihao/Chumor, our project page is at
https://dnaihao.github.io/Chumor-dataset/, our leaderboard is at
https://huggingface.co/spaces/dnaihao/Chumor, and our codebase is at
https://github.com/dnaihao/Chumor-dataset.
 | 2024-12-23T17:19:58Z | [Link](http://arxiv.org/abs/2412.17729v1) |
| Other | Knowledge Editing through Chain-of-Thought |   Large Language Models (LLMs) have demonstrated exceptional capabilities
across a wide range of natural language processing (NLP) tasks. However,
keeping these models up-to-date with evolving world knowledge remains a
significant challenge due to the high costs of frequent retraining. To address
this challenge, knowledge editing techniques have emerged to update LLMs with
new information without rebuilding the model from scratch. Among these, the
in-context editing paradigm stands out for its effectiveness in integrating new
knowledge while preserving the model's original capabilities. Despite its
potential, existing in-context knowledge editing methods are often
task-specific, focusing primarily on multi-hop QA tasks using structured
knowledge triples. Moreover, their reliance on few-shot prompting for task
decomposition makes them unstable and less effective in generalizing across
diverse tasks.
  In response to these limitations, we propose EditCoT, a novel knowledge
editing framework that flexibly and efficiently updates LLMs across various
tasks without retraining. EditCoT works by generating a chain-of-thought (CoT)
for a given input and then iteratively refining this CoT process using a CoT
editor based on updated knowledge. We evaluate EditCoT across a diverse range
of benchmarks, covering multiple languages and tasks. The results demonstrate
that our approach achieves state-of-the-art performance while offering superior
generalization, effectiveness, and stability compared to existing methods,
marking a significant advancement in the field of knowledge updating. Code and
data are available at: https://github.com/bebr2/EditCoT.
 | 2024-12-23T17:17:50Z | [Link](http://arxiv.org/abs/2412.17727v1) |
| Other | RAGONITE: Iterative Retrieval on Induced Databases and Verbalized RDF
  for Conversational QA over KGs with RAG |   Conversational question answering (ConvQA) is a convenient means of searching
over RDF knowledge graphs (KGs), where a prevalent approach is to translate
natural language questions to SPARQL queries. However, SPARQL has certain
shortcomings: (i) it is brittle for complex intents and conversational
questions, and (ii) it is not suitable for more abstract needs. Instead, we
propose a novel two-pronged system where we fuse: (i) SQL-query results over a
database automatically derived from the KG, and (ii) text-search results over
verbalizations of KG facts. Our pipeline supports iterative retrieval: when the
results of any branch are found to be unsatisfactory, the system can
automatically opt for further rounds. We put everything together in a retrieval
augmented generation (RAG) setup, where an LLM generates a coherent response
from accumulated search results. We demonstrate the superiority of our proposed
system over several baselines on a knowledge graph of BMW automobiles.
 | 2024-12-23T16:16:30Z | [Link](http://arxiv.org/abs/2412.17690v1) |
| Other | Generating Completions for Fragmented Broca's Aphasic Sentences Using
  Large Language Models |   Broca's aphasia is a type of aphasia characterized by non-fluent, effortful
and fragmented speech production with relatively good comprehension. Since
traditional aphasia treatment methods are often time-consuming,
labour-intensive, and do not reflect real-world conversations, applying natural
language processing based approaches such as Large Language Models (LLMs) could
potentially contribute to improving existing treatment approaches. To address
this issue, we explore the use of sequence-to-sequence LLMs for completing
fragmented Broca's aphasic sentences. We first generate synthetic Broca's
aphasic data using a rule-based system designed to mirror the linguistic
characteristics of Broca's aphasic speech. Using this synthetic data, we then
fine-tune four pre-trained LLMs on the task of completing fragmented sentences.
We evaluate our fine-tuned models on both synthetic and authentic Broca's
aphasic data. We demonstrate LLMs' capability for reconstructing fragmented
sentences, with the models showing improved performance with longer input
utterances. Our result highlights the LLMs' potential in advancing
communication aids for individuals with Broca's aphasia and possibly other
clinical populations.
 | 2024-12-23T15:54:15Z | [Link](http://arxiv.org/abs/2412.17669v1) |
| Other | SCBench: A Sports Commentary Benchmark for Video LLMs |   Recently, significant advances have been made in Video Large Language Models
(Video LLMs) in both academia and industry. However, methods to evaluate and
benchmark the performance of different Video LLMs, especially their
fine-grained, temporal visual capabilities, remain very limited. On one hand,
current benchmarks use relatively simple videos (e.g., subtitled movie clips)
where the model can understand the entire video by processing just a few
frames. On the other hand, their datasets lack diversity in task format,
comprising only QA or multi-choice QA, which overlooks the models' capacity for
generating in-depth and precise texts. Sports videos, which feature intricate
visual information, sequential events, and emotionally charged commentary,
present a critical challenge for Video LLMs, making sports commentary an ideal
benchmarking task. Inspired by these challenges, we propose a novel task:
sports video commentary generation, developed $\textbf{SCBench}$ for Video
LLMs. To construct such a benchmark, we introduce (1) $\textbf{SCORES}$, a
six-dimensional metric specifically designed for our task, upon which we
propose a GPT-based evaluation method, and (2) $\textbf{CommentarySet}$, a
dataset consisting of 5,775 annotated video clips and ground-truth labels
tailored to our metric. Based on SCBench, we conduct comprehensive evaluations
on multiple Video LLMs (e.g. VILA, Video-LLaVA, etc.) and chain-of-thought
baseline methods. Our results found that InternVL-Chat-2 achieves the best
performance with 5.44, surpassing the second-best by 1.04. Our work provides a
fresh perspective for future research, aiming to enhance models' overall
capabilities in complex visual understanding tasks. Our dataset will be
released soon.
 | 2024-12-23T15:13:56Z | [Link](http://arxiv.org/abs/2412.17637v1) |
| Other | Tracking the Feature Dynamics in LLM Training: A Mechanistic Study |   Understanding training dynamics and feature evolution is crucial for the
mechanistic interpretability of large language models (LLMs). Although sparse
autoencoders (SAEs) have been used to identify features within LLMs, a clear
picture of how these features evolve during training remains elusive. In this
study, we: (1) introduce SAE-Track, a method to efficiently obtain a continual
series of SAEs; (2) formulate the process of feature formation and conduct a
mechanistic analysis; and (3) analyze and visualize feature drift during
training. Our work provides new insights into the dynamics of features in LLMs,
enhancing our understanding of training mechanisms and feature evolution.
 | 2024-12-23T14:58:37Z | [Link](http://arxiv.org/abs/2412.17626v1) |
| Other | Personalized Large Vision-Language Models |   The personalization model has gained significant attention in image
generation yet remains underexplored for large vision-language models (LVLMs).
Beyond generic ones, with personalization, LVLMs handle interactive dialogues
using referential concepts (e.g., ``Mike and Susan are talking.'') instead of
the generic form (e.g., ``a boy and a girl are talking.''), making the
conversation more customizable and referentially friendly. In addition, PLVM is
equipped to continuously add new concepts during a dialogue without incurring
additional costs, which significantly enhances the practicality. PLVM proposes
Aligner, a pre-trained visual encoder to align referential concepts with the
queried images. During the dialogues, it extracts features of reference images
with these corresponding concepts and recognizes them in the queried image,
enabling personalization. We note that the computational cost and parameter
count of the Aligner are negligible within the entire framework. With
comprehensive qualitative and quantitative analyses, we reveal the
effectiveness and superiority of PLVM.
 | 2024-12-23T14:29:41Z | [Link](http://arxiv.org/abs/2412.17610v1) |
| Other | SBS Figures: Pre-training Figure QA from Stage-by-Stage Synthesized
  Images |   Building a large-scale figure QA dataset requires a considerable amount of
work, from gathering and selecting figures to extracting attributes like text,
numbers, and colors, and generating QAs. Although recent developments in LLMs
have led to efforts to synthesize figures, most of these focus primarily on QA
generation. Additionally, creating figures directly using LLMs often encounters
issues such as code errors, similar-looking figures, and repetitive content in
figures. To address this issue, we present SBSFigures (Stage-by-Stage Synthetic
Figures), a dataset for pre-training figure QA. Our proposed pipeline enables
the creation of chart figures with complete annotations of the visualized data
and dense QA annotations without any manual annotation process. Our
stage-by-stage pipeline makes it possible to create diverse topic and
appearance figures efficiently while minimizing code errors. Our SBSFigures
demonstrate a strong pre-training effect, making it possible to achieve
efficient training with a limited amount of real-world chart data starting from
our pre-trained weights.
 | 2024-12-23T14:25:33Z | [Link](http://arxiv.org/abs/2412.17606v1) |
| Other | LiveIdeaBench: Evaluating LLMs' Scientific Creativity and Idea
  Generation with Minimal Context |   While Large Language Models (LLMs) have demonstrated remarkable capabilities
in scientific tasks, existing evaluation frameworks primarily assess their
performance using rich contextual inputs, overlooking their ability to generate
novel ideas from minimal information. We introduce LiveIdeaBench, a
comprehensive benchmark that evaluates LLMs' scientific creativity and
divergent thinking capabilities using single-keyword prompts. Drawing from
Guilford's creativity theory, our framework employs a dynamic panel of
state-of-the-art LLMs to assess generated ideas across four key dimensions:
originality, feasibility, fluency, and flexibility. Through extensive
experimentation with 20 leading models across 1,180 keywords spanning 18
scientific domains, we reveal that scientific creative ability shows distinct
patterns from general intelligence metrics. Notably, our results demonstrate
that models like QwQ-32B-preview achieve comparable creative performance to
top-tier models like o1-preview, despite significant gaps in their general
intelligence scores. These findings highlight the importance of specialized
evaluation frameworks for scientific creativity and suggest that the
development of creative capabilities in LLMs may follow different trajectories
than traditional problem-solving abilities.
 | 2024-12-23T14:13:44Z | [Link](http://arxiv.org/abs/2412.17596v1) |
| Other | Leveraging Memory Retrieval to Enhance LLM-based Generative
  Recommendation |   Leveraging Large Language Models (LLMs) to harness user-item interaction
histories for item generation has emerged as a promising paradigm in generative
recommendation. However, the limited context window of LLMs often restricts
them to focusing on recent user interactions only, leading to the neglect of
long-term interests involved in the longer histories. To address this
challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR),
which is capable of storing long-term interests in the memory and extracting
relevant information from it for next-item generation within LLMs. Extensive
experimental results on two real-world datasets demonstrate the effectiveness
of our proposed AutoMR framework in utilizing long-term interests for
generative recommendation.
 | 2024-12-23T14:10:09Z | [Link](http://arxiv.org/abs/2412.17593v1) |
| Other | GQSA: Group Quantization and Sparsity for Accelerating Large Language
  Model Inference |   With the rapid growth in the scale and complexity of large language models
(LLMs), the costs of training and inference have risen substantially. Model
compression has emerged as a mainstream solution to reduce memory usage and
computational overhead. This paper presents Group Quantization and Sparse
Acceleration (\textbf{GQSA}), a novel compression technique tailored for LLMs.
Traditional methods typically focus exclusively on either quantization or
sparsification, but relying on a single strategy often results in significant
performance loss at high compression rates. In contrast, GQSA integrates
quantization and sparsification in a tightly coupled manner, leveraging
GPU-friendly structured group sparsity and quantization for efficient
acceleration. The proposed method consists of three key steps. First, GQSA
applies group structured pruning to adhere to GPU-friendly sparse pattern
constraints. Second, a two-stage sparsity-aware training process is employed to
maximize performance retention after compression. Finally, the framework adopts
the Block Sparse Row (BSR) format to enable practical deployment and efficient
execution. Experimental results on the LLaMA model family show that GQSA
achieves an excellent balance between model speed and accuracy. Furthermore, on
the latest LLaMA-3 and LLaMA-3.1 models, GQSA outperforms existing LLM
compression techniques significantly.
 | 2024-12-23T13:28:15Z | [Link](http://arxiv.org/abs/2412.17560v1) |
| Other | A Survey of Query Optimization in Large Language Models |   \textit{Query Optimization} (QO) refers to techniques aimed at enhancing the
efficiency and quality of Large Language Models (LLMs) in understanding and
answering queries, especially complex ones in scenarios like
Retrieval-Augmented Generation (RAG). Specifically, RAG mitigates the
limitations of LLMs by dynamically retrieving and leveraging up-to-date
relevant information, which provides a cost-effective solution to the challenge
of LLMs producing plausible but potentially inaccurate responses. Recently, as
RAG evolves and incorporates multiple components that influence its
performance, QO has emerged as a critical element, playing a pivotal role in
determining the effectiveness of RAG's retrieval stage in accurately sourcing
the necessary multiple pieces of evidence to answer queries correctly. In this
paper, we trace the evolution of QO techniques by summarizing and analyzing
significant studies. Through an organized framework and categorization, we aim
to consolidate existing QO techniques in RAG, elucidate their technological
foundations, and highlight their potential to enhance the versatility and
applications of LLMs.
 | 2024-12-23T13:26:04Z | [Link](http://arxiv.org/abs/2412.17558v1) |
| Other | Resource-Aware Arabic LLM Creation: Model Adaptation, Integration, and
  Multi-Domain Testing |   This paper presents a novel approach to fine-tuning the Qwen2-1.5B model for
Arabic language processing using Quantized Low-Rank Adaptation (QLoRA) on a
system with only 4GB VRAM. We detail the process of adapting this large
language model to the Arabic domain, using diverse datasets including Bactrian,
OpenAssistant, and Wikipedia Arabic corpora. Our methodology involves custom
data preprocessing, model configuration, and training optimization techniques
such as gradient accumulation and mixed-precision training. We address specific
challenges in Arabic NLP, including morphological complexity, dialectal
variations, and diacritical mark handling. Experimental results over 10,000
training steps show significant performance improvements, with the final loss
converging to 0.1083. We provide comprehensive analysis of GPU memory usage,
training dynamics, and model evaluation across various Arabic language tasks,
including text classification, question answering, and dialect identification.
The fine-tuned model demonstrates robustness to input perturbations and
improved handling of Arabic-specific linguistic phenomena. This research
contributes to multilingual AI by demonstrating a resource-efficient approach
for creating specialized language models, potentially democratizing access to
advanced NLP technologies for diverse linguistic communities. Our work paves
the way for future research in low-resource language adaptation and efficient
fine-tuning of large language models.
 | 2024-12-23T13:08:48Z | [Link](http://arxiv.org/abs/2412.17548v1) |
| Other | Retention Score: Quantifying Jailbreak Risks for Vision Language Models |   The emergence of Vision-Language Models (VLMs) is a significant advancement
in integrating computer vision with Large Language Models (LLMs) to enhance
multi-modal machine learning capabilities. However, this progress has also made
VLMs vulnerable to sophisticated adversarial attacks, raising concerns about
their reliability. The objective of this paper is to assess the resilience of
VLMs against jailbreak attacks that can compromise model safety compliance and
result in harmful outputs. To evaluate a VLM's ability to maintain its
robustness against adversarial input perturbations, we propose a novel metric
called the \textbf{Retention Score}. Retention Score is a multi-modal
evaluation metric that includes Retention-I and Retention-T scores for
quantifying jailbreak risks in visual and textual components of VLMs. Our
process involves generating synthetic image-text pairs using a conditional
diffusion model. These pairs are then predicted for toxicity score by a VLM
alongside a toxicity judgment classifier. By calculating the margin in toxicity
scores, we can quantify the robustness of the VLM in an attack-agnostic manner.
Our work has four main contributions. First, we prove that Retention Score can
serve as a certified robustness metric. Second, we demonstrate that most VLMs
with visual components are less robust against jailbreak attacks than the
corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find
that the security settings in Google Gemini significantly affect the score and
robustness. Moreover, the robustness of GPT4V is similar to the medium settings
of Gemini. Finally, our approach offers a time-efficient alternative to
existing adversarial attack methods and provides consistent model robustness
rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.
 | 2024-12-23T13:05:51Z | [Link](http://arxiv.org/abs/2412.17544v1) |
| Other | DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM
  Jailbreak |   Large Language Models (LLMs) are susceptible to generating harmful content
when prompted with carefully crafted inputs, a vulnerability known as LLM
jailbreaking. As LLMs become more powerful, studying jailbreak methods is
critical to enhancing security and aligning models with human values.
Traditionally, jailbreak techniques have relied on suffix addition or prompt
templates, but these methods suffer from limited attack diversity. This paper
introduces DiffusionAttacker, an end-to-end generative approach for jailbreak
rewriting inspired by diffusion models. Our method employs a
sequence-to-sequence (seq2seq) text diffusion model as a generator,
conditioning on the original prompt and guiding the denoising process with a
novel attack loss. Unlike previous approaches that use autoregressive LLMs to
generate jailbreak prompts, which limit the modification of already generated
tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq
diffusion model, allowing more flexible token modifications. This approach
preserves the semantic content of the original prompt while producing harmful
content. Additionally, we leverage the Gumbel-Softmax technique to make the
sampling process from the diffusion model's output distribution differentiable,
eliminating the need for iterative token search. Extensive experiments on
Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous
methods across various evaluation metrics, including attack success rate (ASR),
fluency, and diversity.
 | 2024-12-23T12:44:54Z | [Link](http://arxiv.org/abs/2412.17522v1) |
| Other | DRT-o1: Optimized Deep Reasoning Translation via Long Chain-of-Thought |   Recently, O1-like models have emerged as representative examples,
illustrating the effectiveness of long chain-of-thought (CoT) in reasoning
tasks such as math and coding tasks. In this paper, we introduce DRT-o1, an
attempt to bring the success of long CoT to neural machine translation (MT).
Specifically, in view of the literature books that might involve similes and
metaphors, translating these texts to a target language is very difficult in
practice due to cultural differences. In such cases, literal translation often
fails to convey the intended meaning effectively. Even for professional human
translators, considerable thought must be given to preserving semantics
throughout the translation process. To simulate LLMs' long thought ability in
MT, we first mine sentences containing similes or metaphors from existing
literature books, and then develop a multi-agent framework to translate these
sentences via long thought. In the multi-agent framework, a translator is used
to iteratively translate the source sentence under the suggestions provided by
an advisor. To ensure the effectiveness of the long thoughts, an evaluator is
also employed to judge whether the translation in the current round is better
than the previous one or not. In this manner, we collect tens of thousands of
long-thought MT data, which is used to train our DRT-o1. The experimental
results on literature translation demonstrate the effectiveness of the DRT-o1.
Using Qwen2.5-7B and Qwen2.5-14B as the backbones, the improvement brought by
DRT-o1 achieves 7.33~8.26 BLEU and 1.66~3.36 CometScore. Besides, DRT-o1-7B can
outperform QwQ-32B-Preview by 7.82 BLEU and 1.46 CometScore, showing its
effectiveness. The project is available at https://github.com/krystalan/DRT-o1
 | 2024-12-23T11:55:33Z | [Link](http://arxiv.org/abs/2412.17498v1) |
| Other | A Survey on Multi-Generative Agent System: Recent Advances and New
  Frontiers |   Multi-generative agent systems (MGASs) have become a research hotspot since
the rise of large language models (LLMs). However, with the continuous influx
of new related works, the existing reviews struggle to capture them
comprehensively. This paper presents a comprehensive survey of these studies.
We first discuss the definition of MGAS, a framework encompassing much of
previous work. We provide an overview of the various applications of MGAS in
(i) solving complex tasks, (ii) simulating specific scenarios, and (iii)
evaluating generative agents. Building on previous studies, we also highlight
several challenges and propose future directions for research in this field.
 | 2024-12-23T11:11:51Z | [Link](http://arxiv.org/abs/2412.17481v1) |
| Other | Applying LLM and Topic Modelling in Psychotherapeutic Contexts |   This study explores the use of Large language models to analyze therapist
remarks in a psychotherapeutic setting. The paper focuses on the application of
BERTopic, a machine learning-based topic modeling tool, to the dialogue of two
different groups of therapists (classical and modern), which makes it possible
to identify and describe a set of topics that consistently emerge across these
groups. The paper describes in detail the chosen algorithm for BERTopic, which
included creating a vector space from a corpus of therapist remarks, reducing
its dimensionality, clustering the space, and creating and optimizing topic
representation. Along with the automatic topical modeling by the BERTopic, the
research involved an expert assessment of the findings and manual topic
structure optimization. The topic modeling results highlighted the most common
and stable topics in therapists speech, offering insights into how language
patterns in therapy develop and remain stable across different therapeutic
styles. This work contributes to the growing field of machine learning in
psychotherapy by demonstrating the potential of automated methods to improve
both the practice and training of therapists. The study highlights the value of
topic modeling as a tool for gaining a deeper understanding of therapeutic
dialogue and offers new opportunities for improving therapeutic effectiveness
and clinical supervision.
 | 2024-12-23T10:14:32Z | [Link](http://arxiv.org/abs/2412.17449v1) |
| Other | Condor: A Code Discriminator Integrating General Semantics with Code
  Details |   LLMs demonstrate significant potential across various software engineering
tasks. However, they still face challenges in generating correct code on the
first attempt when addressing complex requirements. Introducing a discriminator
to select reliable outputs from multiple generated results is an effective way
to enhance their reliability and stability. Currently, these discriminators
fall into two categories: execution-based discriminators and
non-execution-based discriminators. Execution-based discriminators face
flexibility challenges due to difficulties in obtaining test cases and security
concerns, while non-execution-based discriminators, although more flexible,
struggle to capture subtle differences in code details. To maintain flexibility
while improving the model's ability to capture fine-grained code details, this
paper proposes Condor. We first design contrastive learning to optimize the
code representations of the base model, enabling it to reflect differences in
code details. Then, we leverage intermediate data from the code modification
process to further enrich the discriminator's training data, enhancing its
ability to discern code details. Experimental results indicate that on the
subtle code difference dataset (i.e., CodeNanoFix), Condor significantly
outperforms other discriminators in discriminative performance: Condor (1.3B)
improves the discriminative F1 score of DeepSeek-Coder (1.3B) from 67% to 73%.
In discriminating LLM-generated outputs, Condor (1.3B) and Condor (110M) raise
the Pass@1 score of Meta-Llama-3.1-Instruct (70B) on the CodeNanoFix dataset
from 52.64% to 62.63% and 59.64%, respectively. Moreover, Condor demonstrates
strong generalization capabilities on the MBPP and APPS datasets. For example,
Condor (1.3B) improves the Pass@1 of Meta-Llama-3.1-Instruct (70B) on the APPS
dataset by 147.05%.
 | 2024-12-23T09:47:20Z | [Link](http://arxiv.org/abs/2412.17429v1) |
| Other | VidCtx: Context-aware Video Question Answering with Image Models |   To address computational and memory limitations of Large Multimodal Models in
the Video Question-Answering task, several recent methods extract textual
representations per frame (e.g., by captioning) and feed them to a Large
Language Model (LLM) that processes them to produce the final response.
However, in this way, the LLM does not have access to visual information and
often has to process repetitive textual descriptions of nearby frames. To
address those shortcomings, in this paper, we introduce VidCtx, a novel
training-free VideoQA framework which integrates both modalities, i.e. both
visual information from input frames and textual descriptions of others frames
that give the appropriate context. More specifically, in the proposed framework
a pre-trained Large Multimodal Model (LMM) is prompted to extract at regular
intervals, question-aware textual descriptions (captions) of video frames.
Those will be used as context when the same LMM will be prompted to answer the
question at hand given as input a) a certain frame, b) the question and c) the
context/caption of an appropriate frame. To avoid redundant information, we
chose as context the descriptions of distant frames. Finally, a simple yet
effective max pooling mechanism is used to aggregate the frame-level decisions.
This methodology enables the model to focus on the relevant segments of the
video and scale to a high number of frames. Experiments show that VidCtx
achieves competitive performance among approaches that rely on open models on
three public Video QA benchmarks, NExT-QA, IntentQA and STAR.
 | 2024-12-23T09:26:38Z | [Link](http://arxiv.org/abs/2412.17415v1) |
| Other | Just What You Desire: Constrained Timeline Summarization with
  Self-Reflection for Enhanced Relevance |   Given news articles about an entity, such as a public figure or organization,
timeline summarization (TLS) involves generating a timeline that summarizes the
key events about the entity. However, the TLS task is too underspecified, since
what is of interest to each reader may vary, and hence there is not a single
ideal or optimal timeline. In this paper, we introduce a novel task, called
Constrained Timeline Summarization (CTLS), where a timeline is generated in
which all events in the timeline meet some constraint. An example of a
constrained timeline concerns the legal battles of Tiger Woods, where only
events related to his legal problems are selected to appear in the timeline. We
collected a new human-verified dataset of constrained timelines involving 47
entities and 5 constraints per entity. We propose an approach that employs a
large language model (LLM) to summarize news articles according to a specified
constraint and cluster them to identify key events to include in a constrained
timeline. In addition, we propose a novel self-reflection method during summary
generation, demonstrating that this approach successfully leads to improved
performance.
 | 2024-12-23T09:17:06Z | [Link](http://arxiv.org/abs/2412.17408v1) |
| Other | Towards Intrinsic Self-Correction Enhancement in Monte Carlo Tree Search
  Boosted Reasoning via Iterative Preference Learning |   With current state-of-the-art approaches aimed at enhancing the reasoning
capabilities of Large Language Models(LLMs) through iterative preference
learning inspired by AlphaZero, we propose to further enhance the step-wise
reasoning capabilities through intrinsic self-correction to some extent. Our
work leverages step-wise preference learning to enhance self-verification via
reinforcement learning. We initially conduct our work through a two-stage
training procedure. At the first stage, the self-correction reasoning ability
of an LLM is enhanced through its own predictions, relying entirely on
self-generated data within the intrinsic self-correction to some extent. At the
second stage, the baseline step-wise preference learning is leveraged via the
application of the enhanced self-correct policy achieved at the first stage. In
the evaluation of arithmetic reasoning tasks, our approach outperforms
OpenMath2-Llama3.1-8B, dart-math-mistral-7b-uniform on MATH with increases in
accuracy to 71.34%(+4.18%) and 48.06%(+4.94%) and LLama-3.1-8B-Instruct,
Mistral-7B-Instruct-v0.1 on GSM8K with increases in accuracy to 86.76%(+2.00%)
and 38.06%(+2.28%).
 | 2024-12-23T08:51:48Z | [Link](http://arxiv.org/abs/2412.17397v1) |
| Other | WarriorCoder: Learning from Expert Battles to Augment Code Large
  Language Models |   Despite recent progress achieved by code large language models (LLMs), their
remarkable abilities are largely dependent on fine-tuning on the high-quality
data, posing challenges for data collection and annotation. To address this,
current methods often design various data flywheels to gather complex code
instructions, enabling models to handle more intricate tasks. However, these
approaches typically rely on off-the-shelf datasets and data augmentation from
the limited pool of proprietary LLMs (e.g., Claude, GPT4, and so on), which
limits the diversity of the constructed data and makes it prone to systemic
biases. In this paper, we propose WarriorCoder which learns from expert battles
to address these limitations. Specifically, we create an arena for current
expert code LLMs, where each model challenges and responds to others'
challenges, with evaluations conducted by uninvolved judge models. This
competitive framework generates novel training data constructed from scratch,
harnessing the strengths of all participants. Experimental results demonstrate
that WarriorCoder achieves competitive performance compared to previous
methods, even without relying on proprietary LLMs.
 | 2024-12-23T08:47:42Z | [Link](http://arxiv.org/abs/2412.17395v1) |
| Other | Interweaving Memories of a Siamese Large Language Model |   Parameter-efficient fine-tuning (PEFT) methods optimize large language models
(LLMs) by modifying or introducing a small number of parameters to enhance
alignment with downstream tasks. However, they can result in catastrophic
forgetting, where LLMs prioritize new knowledge at the expense of comprehensive
world knowledge. A promising approach to mitigate this issue is to recall prior
memories based on the original knowledge. To this end, we propose a
model-agnostic PEFT framework, IMSM, which Interweaves Memories of a Siamese
Large Language Model. Specifically, our siamese LLM is equipped with an
existing PEFT method. Given an incoming query, it generates two distinct
memories based on the pre-trained and fine-tuned parameters. IMSM then
incorporates an interweaving mechanism that regulates the contributions of both
original and enhanced memories when generating the next token. This framework
is theoretically applicable to all open-source LLMs and existing PEFT methods.
We conduct extensive experiments across various benchmark datasets, evaluating
the performance of popular open-source LLMs using the proposed IMSM, in
comparison to both classical and leading PEFT methods. Our findings indicate
that IMSM maintains comparable time and space efficiency to backbone PEFT
methods while significantly improving performance and effectively mitigating
catastrophic forgetting.
 | 2024-12-23T08:33:47Z | [Link](http://arxiv.org/abs/2412.17383v1) |
| Other | Boosting LLM via Learning from Data Iteratively and Selectively |   Datasets nowadays are generally constructed from multiple sources and using
different synthetic techniques, making data de-noising and de-duplication
crucial before being used for post-training. In this work, we propose to
perform instruction tuning by iterative data selection (\ApproachName{}). We
measure the quality of a sample from complexity and diversity simultaneously.
Instead of calculating the complexity score once for all before fine-tuning, we
highlight the importance of updating this model-specific score during
fine-tuning to accurately accommodate the dynamic changes of the model. On the
other hand, the diversity score is defined on top of the samples' responses
under the consideration of their informativeness. IterIT integrates the
strengths of both worlds by iteratively updating the complexity score for the
top-ranked samples and greedily selecting the ones with the highest
complexity-diversity score. Experiments on multiple instruction-tuning data
demonstrate consistent improvements of IterIT over strong baselines. Moreover,
our approach also generalizes well to domain-specific scenarios and different
backbone models. All resources will be available at
https://github.com/JiaQiSJTU/IterIT.
 | 2024-12-23T08:01:24Z | [Link](http://arxiv.org/abs/2412.17365v1) |
| Other | A Dual-Perspective Metaphor Detection Framework Using Large Language
  Models |   Metaphor detection, a critical task in natural language processing, involves
identifying whether a particular word in a sentence is used metaphorically.
Traditional approaches often rely on supervised learning models that implicitly
encode semantic relationships based on metaphor theories. However, these
methods often suffer from a lack of transparency in their decision-making
processes, which undermines the reliability of their predictions. Recent
research indicates that LLMs (large language models) exhibit significant
potential in metaphor detection. Nevertheless, their reasoning capabilities are
constrained by predefined knowledge graphs. To overcome these limitations, we
propose DMD, a novel dual-perspective framework that harnesses both implicit
and explicit applications of metaphor theories to guide LLMs in metaphor
detection and adopts a self-judgment mechanism to validate the responses from
the aforementioned forms of guidance. In comparison to previous methods, our
framework offers more transparent reasoning processes and delivers more
reliable predictions. Experimental results prove the effectiveness of DMD,
demonstrating state-of-the-art performance across widely-used datasets.
 | 2024-12-23T06:50:04Z | [Link](http://arxiv.org/abs/2412.17332v1) |
| Other | Assessing Human Editing Effort on LLM-Generated Texts via
  Compression-Based Edit Distance |   Assessing the extent of human edits on texts generated by Large Language
Models (LLMs) is crucial to understanding the human-AI interactions and
improving the quality of automated text generation systems. Existing edit
distance metrics, such as Levenshtein, BLEU, ROUGE, and TER, often fail to
accurately measure the effort required for post-editing, especially when edits
involve substantial modifications, such as block operations. In this paper, we
introduce a novel compression-based edit distance metric grounded in the
Lempel-Ziv-77 algorithm, designed to quantify the amount of post-editing
applied to LLM-generated texts. Our method leverages the properties of text
compression to measure the informational difference between the original and
edited texts. Through experiments on real-world human edits datasets, we
demonstrate that our proposed metric is highly correlated with actual edit time
and effort. We also show that LLMs exhibit an implicit understanding of editing
speed, that aligns well with our metric. Furthermore, we compare our metric
with existing ones, highlighting its advantages in capturing complex edits with
linear computational efficiency. Our code and data are available at:
https://github.com/NDV-tiime/CompressionDistance
 | 2024-12-23T06:29:25Z | [Link](http://arxiv.org/abs/2412.17321v1) |
| Other | CodeV: Issue Resolving with Visual Data |   Large Language Models (LLMs) have advanced rapidly in recent years, with
their applications in software engineering expanding to more complex
repository-level tasks. GitHub issue resolving is a key challenge among these
tasks. While recent approaches have made progress on this task, they focus on
textual data within issues, neglecting visual data. However, this visual data
is crucial for resolving issues as it conveys additional knowledge that text
alone cannot. We propose CodeV, the first approach to leveraging visual data to
enhance the issue-resolving capabilities of LLMs. CodeV resolves each issue by
following a two-phase process: data processing and patch generation. To
evaluate CodeV, we construct a benchmark for visual issue resolving, namely
Visual SWE-bench. Through extensive experiments, we demonstrate the
effectiveness of CodeV, as well as provide valuable insights into leveraging
visual data to resolve GitHub issues.
 | 2024-12-23T06:17:11Z | [Link](http://arxiv.org/abs/2412.17315v1) |
| Other | On the Feasibility of Vision-Language Models for Time-Series
  Classification |   We build upon time-series classification by leveraging the capabilities of
Vision Language Models (VLMs). We find that VLMs produce competitive results
after two or less epochs of fine-tuning. We develop a novel approach that
incorporates graphical data representations as images in conjunction with
numerical data. This approach is rooted in the hypothesis that graphical
representations can provide additional contextual information that numerical
data alone may not capture. Additionally, providing a graphical representation
can circumvent issues such as limited context length faced by LLMs. To further
advance this work, we implemented a scalable end-to-end pipeline for training
on different scenarios, allowing us to isolate the most effective strategies
for transferring learning capabilities from LLMs to Time Series Classification
(TSC) tasks. Our approach works with univariate and multivariate time-series
data. In addition, we conduct extensive and practical experiments to show how
this approach works for time-series classification and generative labels.
 | 2024-12-23T05:52:17Z | [Link](http://arxiv.org/abs/2412.17304v1) |
| Other | Prompting in the Wild: An Empirical Study of Prompt Evolution in
  Software Repositories |   The adoption of Large Language Models (LLMs) is reshaping software
development as developers integrate these LLMs into their applications. In such
applications, prompts serve as the primary means of interacting with LLMs.
Despite the widespread use of LLM-integrated applications, there is limited
understanding of how developers manage and evolve prompts. This study presents
the first empirical analysis of prompt evolution in LLM-integrated software
development. We analyzed 1,262 prompt changes across 243 GitHub repositories to
investigate the patterns and frequencies of prompt changes, their relationship
with code changes, documentation practices, and their impact on system
behavior. Our findings show that developers primarily evolve prompts through
additions and modifications, with most changes occurring during feature
development. We identified key challenges in prompt engineering: only 21.9\% of
prompt changes are documented in commit messages, changes can introduce logical
inconsistencies, and misalignment often occurs between prompt changes and LLM
responses. These insights emphasize the need for specialized testing
frameworks, automated validation tools, and improved documentation practices to
enhance the reliability of LLM-integrated applications.
 | 2024-12-23T05:41:01Z | [Link](http://arxiv.org/abs/2412.17298v1) |
| Other | AV-EmoDialog: Chat with Audio-Visual Users Leveraging Emotional Cues |   In human communication, both verbal and non-verbal cues play a crucial role
in conveying emotions, intentions, and meaning beyond words alone. These
non-linguistic information, such as facial expressions, eye contact, voice
tone, and pitch, are fundamental elements of effective interactions, enriching
conversations by adding emotional and contextual depth. Recognizing the
importance of non-linguistic content in communication, we present AV-EmoDialog,
a dialogue system designed to exploit verbal and non-verbal information from
users' audio-visual inputs to generate more responsive and empathetic
interactions. AV-EmoDialog systematically exploits the emotional cues in
audio-visual dialogues; extracting speech content and emotional tones from
speech, analyzing fine-grained facial expressions from visuals, and integrating
these cues to generate emotionally aware responses in an end-to-end manner.
Through extensive experiments, we validate that the proposed AV-EmoDialog
outperforms existing multimodal LLMs in generating not only emotionally
appropriate but also contextually appropriate responses.
 | 2024-12-23T05:24:26Z | [Link](http://arxiv.org/abs/2412.17292v1) |
| Other | Multi-Modal Grounded Planning and Efficient Replanning For Learning
  Embodied Agents with A Few Examples |   Learning a perception and reasoning module for robotic assistants to plan
steps to perform complex tasks based on natural language instructions often
requires large free-form language annotations, especially for short high-level
instructions. To reduce the cost of annotation, large language models (LLMs)
are used as a planner with few data. However, when elaborating the steps, even
the state-of-the-art planner that uses LLMs mostly relies on linguistic common
sense, often neglecting the status of the environment at command reception,
resulting in inappropriate plans. To generate plans grounded in the
environment, we propose FLARE (Few-shot Language with environmental Adaptive
Replanning Embodied agent), which improves task planning using both language
command and environmental perception. As language instructions often contain
ambiguities or incorrect expressions, we additionally propose to correct the
mistakes using visual cues from the agent. The proposed scheme allows us to use
a few language pairs thanks to the visual cues and outperforms state-of-the-art
approaches. Our code is available at https://github.com/snumprlab/flare.
 | 2024-12-23T05:20:01Z | [Link](http://arxiv.org/abs/2412.17288v1) |
| Other | LLM4AD: A Platform for Algorithm Design with Large Language Model |   We introduce LLM4AD, a unified Python platform for algorithm design (AD) with
large language models (LLMs). LLM4AD is a generic framework with modularized
blocks for search methods, algorithm design tasks, and LLM interface. The
platform integrates numerous key methods and supports a wide range of algorithm
design tasks across various domains including optimization, machine learning,
and scientific discovery. We have also designed a unified evaluation sandbox to
ensure a secure and robust assessment of algorithms. Additionally, we have
compiled a comprehensive suite of support resources, including tutorials,
examples, a user manual, online resources, and a dedicated graphical user
interface (GUI) to enhance the usage of LLM4AD. We believe this platform will
serve as a valuable tool for fostering future development in the merging
research direction of LLM-assisted algorithm design.
 | 2024-12-23T05:12:54Z | [Link](http://arxiv.org/abs/2412.17287v1) |
| Other | LegalAgentBench: Evaluating LLM Agents in Legal Domain |   With the increasing intelligence and autonomy of LLM agents, their potential
applications in the legal domain are becoming increasingly apparent. However,
existing general-domain benchmarks cannot fully capture the complexity and
subtle nuances of real-world judicial cognition and decision-making. Therefore,
we propose LegalAgentBench, a comprehensive benchmark specifically designed to
evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17
corpora from real-world legal scenarios and provides 37 tools for interacting
with external knowledge. We designed a scalable task construction framework and
carefully annotated 300 tasks. These tasks span various types, including
multi-hop reasoning and writing, and range across different difficulty levels,
effectively reflecting the complexity of real-world legal scenarios. Moreover,
beyond evaluating final success, LegalAgentBench incorporates keyword analysis
during intermediate processes to calculate progress rates, enabling more
fine-grained evaluation. We evaluated eight popular LLMs, highlighting the
strengths, limitations, and potential areas for improvement of existing models
and methods. LegalAgentBench sets a new benchmark for the practical application
of LLMs in the legal domain, with its code and data available at
\url{https://github.com/CSHaitao/LegalAgentBench}.
 | 2024-12-23T04:02:46Z | [Link](http://arxiv.org/abs/2412.17259v1) |
| Other | Unlocking Cross-Lingual Sentiment Analysis through Emoji Interpretation:
  A Multimodal Generative AI Approach |   Emojis have become ubiquitous in online communication, serving as a universal
medium to convey emotions and decorative elements. Their widespread use
transcends language and cultural barriers, enhancing understanding and
fostering more inclusive interactions. While existing work gained valuable
insight into emojis understanding, exploring emojis' capability to serve as a
universal sentiment indicator leveraging large language models (LLMs) has not
been thoroughly examined. Our study aims to investigate the capacity of emojis
to serve as reliable sentiment markers through LLMs across languages and
cultures. We leveraged the multimodal capabilities of ChatGPT to explore the
sentiments of various representations of emojis and evaluated how well
emoji-conveyed sentiment aligned with text sentiment on a multi-lingual dataset
collected from 32 countries. Our analysis reveals that the accuracy of
LLM-based emoji-conveyed sentiment is 81.43%, underscoring emojis' significant
potential to serve as a universal sentiment marker. We also found a consistent
trend that the accuracy of sentiment conveyed by emojis increased as the number
of emojis grew in text. The results reinforce the potential of emojis to serve
as global sentiment indicators, offering insight into fields such as
cross-lingual and cross-cultural sentiment analysis on social media platforms.
Code: https://github.com/ResponsibleAILab/emoji-universal-sentiment.
 | 2024-12-23T03:57:45Z | [Link](http://arxiv.org/abs/2412.17255v1) |
| Other | SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval |   The performance of Dense retrieval (DR) is significantly influenced by the
quality of negative sampling. Traditional DR methods primarily depend on naive
negative sampling techniques or on mining hard negatives through external
retriever and meticulously crafted strategies. However, naive negative sampling
often fails to adequately capture the accurate boundaries between positive and
negative samples, whereas existing hard negative sampling methods are prone to
false negatives, resulting in performance degradation and training instability.
Recent advancements in large language models (LLMs) offer an innovative
solution to these challenges by generating contextually rich and diverse
negative samples. In this work, we present a framework that harnesses LLMs to
synthesize high-quality hard negative samples. We first devise a
\textit{multi-attribute self-reflection prompting strategy} to direct LLMs in
hard negative sample generation. Then, we implement a \textit{hybrid sampling
strategy} that integrates these synthetic negatives with traditionally
retrieved negatives, thereby stabilizing the training process and improving
retrieval performance. Extensive experiments on five benchmark datasets
demonstrate the efficacy of our approach, and code is also publicly available.
 | 2024-12-23T03:49:00Z | [Link](http://arxiv.org/abs/2412.17250v1) |
| Other | EM-MIAs: Enhancing Membership Inference Attacks in Large Language Models
  through Ensemble Modeling |   With the widespread application of large language models (LLM), concerns
about the privacy leakage of model training data have increasingly become a
focus. Membership Inference Attacks (MIAs) have emerged as a critical tool for
evaluating the privacy risks associated with these models. Although existing
attack methods, such as LOSS, Reference-based, min-k, and zlib, perform well in
certain scenarios, their effectiveness on large pre-trained language models
often approaches random guessing, particularly in the context of large-scale
datasets and single-epoch training. To address this issue, this paper proposes
a novel ensemble attack method that integrates several existing MIAs techniques
(LOSS, Reference-based, min-k, zlib) into an XGBoost-based model to enhance
overall attack performance (EM-MIAs). Experimental results demonstrate that the
ensemble model significantly improves both AUC-ROC and accuracy compared to
individual attack methods across various large language models and datasets.
This indicates that by combining the strengths of different methods, we can
more effectively identify members of the model's training data, thereby
providing a more robust tool for evaluating the privacy risks of LLM. This
study offers new directions for further research in the field of LLM privacy
protection and underscores the necessity of developing more powerful privacy
auditing methods.
 | 2024-12-23T03:47:54Z | [Link](http://arxiv.org/abs/2412.17249v1) |
| Other | On the Generalization Ability of Machine-Generated Text Detectors |   The rise of large language models (LLMs) has raised concerns about
machine-generated text (MGT), including ethical and practical issues like
plagiarism and misinformation. Building a robust and highly generalizable MGT
detection system has become increasingly important. This work investigates the
generalization capabilities of MGT detectors in three aspects: First, we
construct MGTAcademic, a large-scale dataset focused on academic writing,
featuring human-written texts (HWTs) and MGTs across STEM, Humanities, and
Social Sciences, paired with an extensible code framework for efficient
benchmarking. Second, we investigate the transferability of detectors across
domains and LLMs, leveraging fine-grained datasets to reveal insights into
domain transferring and implementing few-shot techniques to improve the
performance by roughly 13.2%. Third, we introduce a novel attribution task
where models must adapt to new classes over time without (or with very limited)
access to prior training data and benchmark detectors. We implement several
adapting techniques to improve the performance by roughly 10% and highlight the
inherent complexity of the task. Our findings provide insights into the
generalization ability of MGT detectors across diverse scenarios and lay the
foundation for building robust, adaptive detection systems.
 | 2024-12-23T03:30:34Z | [Link](http://arxiv.org/abs/2412.17242v1) |
| Other | Better Think with Tables: Leveraging Tables to Enhance Large Language
  Model Comprehension |   Despite the recent advancement of Large Langauge Models (LLMs), they struggle
with complex queries often involving multiple conditions, common in real-world
scenarios. We propose Thinking with Tables, a technique that assists LLMs to
leverage tables for intermediate thinking aligning with human cognitive
behavior. By introducing a pre-instruction that triggers an LLM to organize
information in tables, our approach achieves a 40.29\% average relative
performance increase, higher robustness, and show generalizability to different
requests, conditions, or scenarios. We additionally show the influence of data
structuredness for the model by comparing results from four distinct
structuring levels that we introduce.
 | 2024-12-22T23:31:03Z | [Link](http://arxiv.org/abs/2412.17189v1) |
| Other | Enhancing Item Tokenization for Generative Recommendation through
  Self-Improvement |   Generative recommendation systems, driven by large language models (LLMs),
present an innovative approach to predicting user preferences by modeling items
as token sequences and generating recommendations in a generative manner. A
critical challenge in this approach is the effective tokenization of items,
ensuring that they are represented in a form compatible with LLMs. Current item
tokenization methods include using text descriptions, numerical strings, or
sequences of discrete tokens. While text-based representations integrate
seamlessly with LLM tokenization, they are often too lengthy, leading to
inefficiencies and complicating accurate generation. Numerical strings, while
concise, lack semantic depth and fail to capture meaningful item relationships.
Tokenizing items as sequences of newly defined tokens has gained traction, but
it often requires external models or algorithms for token assignment. These
external processes may not align with the LLM's internal pretrained
tokenization schema, leading to inconsistencies and reduced model performance.
To address these limitations, we propose a self-improving item tokenization
method that allows the LLM to refine its own item tokenizations during training
process. Our approach starts with item tokenizations generated by any external
model and periodically adjusts these tokenizations based on the LLM's learned
patterns. Such alignment process ensures consistency between the tokenization
and the LLM's internal understanding of the items, leading to more accurate
recommendations. Furthermore, our method is simple to implement and can be
integrated as a plug-and-play enhancement into existing generative
recommendation systems. Experimental results on multiple datasets and using
various initial tokenization strategies demonstrate the effectiveness of our
method, with an average improvement of 8\% in recommendation performance.
 | 2024-12-22T21:56:15Z | [Link](http://arxiv.org/abs/2412.17171v1) |
| Other | LLM-based relevance assessment still can't replace human relevance
  assessment |   The use of large language models (LLMs) for relevance assessment in
information retrieval has gained significant attention, with recent studies
suggesting that LLM-based judgments provide comparable evaluations to human
judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim
that LLM-based relevance assessments, such as those generated by the UMBRELA
system, can fully replace traditional human relevance assessments in TREC-style
evaluations. This paper critically examines this claim, highlighting practical
and theoretical limitations that undermine the validity of this conclusion.
First, we question whether the evidence provided by Upadhyay et al. really
supports their claim, particularly if a test collection is used asa benchmark
for future improvements. Second, through a submission deliberately intended to
do so, we demonstrate the ease with which automatic evaluation metrics can be
subverted, showing that systems designed to exploit these evaluations can
achieve artificially high scores. Theoretical challenges -- such as the
inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and
the potential degradation of future LLM performance -- must be addressed before
LLM-based relevance assessments can be considered a viable replacement for
human judgments.
 | 2024-12-22T20:45:15Z | [Link](http://arxiv.org/abs/2412.17156v1) |
| Other | A Multi-AI Agent System for Autonomous Optimization of Agentic AI
  Solutions via Iterative Refinement and LLM-Driven Feedback Loops |   Agentic AI systems use specialized agents to handle tasks within complex
workflows, enabling automation and efficiency. However, optimizing these
systems often requires labor-intensive, manual adjustments to refine roles,
tasks, and interactions. This paper introduces a framework for autonomously
optimizing Agentic AI solutions across industries, such as NLP-driven
enterprise applications. The system employs agents for Refinement, Execution,
Evaluation, Modification, and Documentation, leveraging iterative feedback
loops powered by an LLM (Llama 3.2-3B). The framework achieves optimal
performance without human input by autonomously generating and testing
hypotheses to improve system configurations. This approach enhances scalability
and adaptability, offering a robust solution for real-world applications in
dynamic environments. Case studies across diverse domains illustrate the
transformative impact of this framework, showcasing significant improvements in
output quality, relevance, and actionability. All data for these case studies,
including original and evolved agent codes, along with their outputs, are here:
https://anonymous.4open.science/r/evolver-1D11/
 | 2024-12-22T20:08:04Z | [Link](http://arxiv.org/abs/2412.17149v1) |
| Other | LLM Agent for Fire Dynamics Simulations |   Significant advances have been achieved in leveraging foundation models, such
as large language models (LLMs), to accelerate complex scientific workflows. In
this work we introduce FoamPilot, a proof-of-concept LLM agent designed to
enhance the usability of FireFOAM, a specialized solver for fire dynamics and
fire suppression simulations built using OpenFOAM, a popular open-source
toolbox for computational fluid dynamics (CFD). FoamPilot provides three core
functionalities: code insight, case configuration and simulation evaluation.
Code insight is an alternative to traditional keyword searching leveraging
retrieval-augmented generation (RAG) and aims to enable efficient navigation
and summarization of the FireFOAM source code for developers and experienced
users. For case configuration, the agent interprets user requests in natural
language and aims to modify existing simulation setups accordingly to support
intermediate users. FoamPilot's job execution functionality seeks to manage the
submission and execution of simulations in high-performance computing (HPC)
environments and provide preliminary analysis of simulation results to support
less experienced users. Promising results were achieved for each functionality,
particularly for simple tasks, and opportunities were identified for
significant further improvement for more complex tasks. The integration of
these functionalities into a single LLM agent is a step aimed at accelerating
the simulation workflow for engineers and scientists employing FireFOAM for
complex simulations critical for improving fire safety.
 | 2024-12-22T20:03:35Z | [Link](http://arxiv.org/abs/2412.17146v1) |
| Other | Hate Speech Detection and Target Identification in Devanagari Languages
  via Parameter Efficient Fine-Tuning of LLMs |   The detection of hate speech has become increasingly important in combating
online hostility and its real-world consequences. Despite recent advancements,
there is limited research addressing hate speech detection in
Devanagari-scripted languages, where resources and tools are scarce. While
large language models (LLMs) have shown promise in language-related tasks,
traditional fine-tuning approaches are often infeasible given the size of the
models. In this paper, we propose a Parameter Efficient Fine tuning (PEFT)
based solution for hate speech detection and target identification. We evaluate
multiple LLMs on the Devanagari dataset provided by (Thapa et al., 2025), which
contains annotated instances in 2 languages - Hindi and Nepali. The results
demonstrate the efficacy of our approach in handling Devanagari-scripted
content.
 | 2024-12-22T18:38:24Z | [Link](http://arxiv.org/abs/2412.17131v1) |
| Other | Lies, Damned Lies, and Distributional Language Statistics: Persuasion
  and Deception with Large Language Models |   Large Language Models (LLMs) can generate content that is as persuasive as
human-written text and appear capable of selectively producing deceptive
outputs. These capabilities raise concerns about potential misuse and
unintended consequences as these systems become more widely deployed. This
review synthesizes recent empirical work examining LLMs' capacity and
proclivity for persuasion and deception, analyzes theoretical risks that could
arise from these capabilities, and evaluates proposed mitigations. While
current persuasive effects are relatively small, various mechanisms could
increase their impact, including fine-tuning, multimodality, and social
factors. We outline key open questions for future research, including how
persuasive AI systems might become, whether truth enjoys an inherent advantage
over falsehoods, and how effective different mitigation strategies may be in
practice.
 | 2024-12-22T18:34:10Z | [Link](http://arxiv.org/abs/2412.17128v1) |
| Other | DreamOmni: Unified Image Generation and Editing |   Currently, the success of large language models (LLMs) illustrates that a
unified multitasking approach can significantly enhance model usability,
streamline deployment, and foster synergistic benefits across different tasks.
However, in computer vision, while text-to-image (T2I) models have
significantly improved generation quality through scaling up, their framework
design did not initially consider how to unify with downstream tasks, such as
various types of editing. To address this, we introduce DreamOmni, a unified
model for image generation and editing. We begin by analyzing existing
frameworks and the requirements of downstream tasks, proposing a unified
framework that integrates both T2I models and various editing tasks.
Furthermore, another key challenge is the efficient creation of high-quality
editing data, particularly for instruction-based and drag-based editing. To
this end, we develop a synthetic data pipeline using sticker-like elements to
synthesize accurate, high-quality datasets efficiently, which enables editing
data scaling up for unified model training. For training, DreamOmni jointly
trains T2I generation and downstream tasks. T2I training enhances the model's
understanding of specific concepts and improves generation quality, while
editing training helps the model grasp the nuances of the editing task. This
collaboration significantly boosts editing performance. Extensive experiments
confirm the effectiveness of DreamOmni. The code and model will be released.
 | 2024-12-22T17:17:28Z | [Link](http://arxiv.org/abs/2412.17098v1) |
| Other | Analysis on LLMs Performance for Code Summarization |   Code summarization aims to generate concise natural language descriptions for
source code. Deep learning has been used more and more recently in software
engineering, particularly for tasks like code creation and summarization.
Specifically, it appears that the most current Large Language Models with
coding perform well on these tasks. Large Language Models (LLMs) have
significantly advanced the field of code summarization, providing sophisticated
methods for generating concise and accurate summaries of source code. This
study aims to perform a comparative analysis of several open-source LLMs,
namely LLaMA-3, Phi-3, Mistral, and Gemma. These models' performance is
assessed using important metrics such as BLEU\textsubscript{3.1} and
ROUGE\textsubscript{3.2}.
  Through this analysis, we seek to identify the strengths and weaknesses of
each model, offering insights into their applicability and effectiveness in
code summarization tasks. Our findings contribute to the ongoing development
and refinement of LLMs, supporting their integration into tools that enhance
software development and maintenance processes.
 | 2024-12-22T17:09:34Z | [Link](http://arxiv.org/abs/2412.17094v1) |
| Other | SAIL: Sample-Centric In-Context Learning for Document Information
  Extraction |   Document Information Extraction (DIE) aims to extract structured information
from Visually Rich Documents (VRDs). Previous full-training approaches have
demonstrated strong performance but may struggle with generalization to unseen
data. In contrast, training-free methods leverage powerful pre-trained models
like Large Language Models (LLMs) to address various downstream tasks with only
a few examples. Nonetheless, training-free methods for DIE encounter two
primary challenges: (1) understanding the complex relationship between layout
and textual elements in VRDs, and (2) providing accurate guidance to
pre-trained models. To address these challenges, we propose Sample-centric
In-context Learning (SAIL) for DIE. SAIL introduces a fine-grained entity-level
textual similarity to facilitate in-depth text analysis by LLMs and
incorporates layout similarity to enhance the analysis of layouts in VRDs.
Additionally, SAIL formulates a unified In-Context Learning (ICL) prompt
template for various sample-centric examples, enabling tailored prompts that
deliver precise guidance to pre-trained models for each sample. Extensive
experiments on FUNSD, CORD, and SROIE benchmarks with various base models
(e.g., LLMs) indicate that our method outperforms training-free baselines, even
closer to the full-training methods. The results show the superiority and
generalization of our method.
 | 2024-12-22T16:58:59Z | [Link](http://arxiv.org/abs/2412.17092v1) |
| Other | The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG
  Applications Using an LLM's Internal States |   Detecting hallucinations in large language models (LLMs) is critical for
enhancing their reliability and trustworthiness. Most research focuses on
hallucinations as deviations from information seen during training. However,
the opaque nature of an LLM's parametric knowledge complicates the
understanding of why generated texts appear ungrounded: The LLM might not have
picked up the necessary knowledge from large and often inaccessible datasets,
or the information might have been changed or contradicted during further
training. Our focus is on hallucinations involving information not used in
training, which we determine by using recency to ensure the information emerged
after a cut-off date. This study investigates these hallucinations by detecting
them at sentence level using different internal states of various LLMs. We
present HalluRAG, a dataset designed to train classifiers on these
hallucinations. Depending on the model and quantization, MLPs trained on
HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with
Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results
show that IAVs detect hallucinations as effectively as CEVs and reveal that
answerable and unanswerable prompts are encoded differently as separate
classifiers for these categories improved accuracy. However, HalluRAG showed
some limited generalizability, advocating for more diversity in datasets on
hallucinations.
 | 2024-12-22T15:08:24Z | [Link](http://arxiv.org/abs/2412.17056v1) |
| Other | DR-Encoder: Encode Low-rank Gradients with Random Prior for Large
  Language Models Differentially Privately |   The emergence of the Large Language Model (LLM) has shown their superiority
in a wide range of disciplines, including language understanding and
translation, relational logic reasoning, and even partial differential
equations solving. The transformer is the pervasive backbone architecture for
the foundation model construction. It is vital to research how to adjust the
Transformer architecture to achieve an end-to-end privacy guarantee in LLM
fine-tuning. In this paper, we investigate three potential information leakage
during a federated fine-tuning procedure for LLM (FedLLM). Based on the
potential information leakage, we provide an end-to-end privacy guarantee
solution for FedLLM by inserting two-stage randomness. The first stage is to
train a gradient auto-encoder with a Gaussian random prior based on the
statistical information of the gradients generated by local clients. The second
stage is to fine-tune the overall LLM with a differential privacy guarantee by
adopting appropriate Gaussian noises. We show the efficiency and accuracy gains
of our proposed method with several foundation models and two popular
evaluation benchmarks. Furthermore, we present a comprehensive privacy analysis
with Gaussian Differential Privacy (GDP) and Renyi Differential Privacy (RDP).
 | 2024-12-22T15:06:09Z | [Link](http://arxiv.org/abs/2412.17053v1) |
| Other | ViLBias: A Framework for Bias Detection using Linguistic and Visual Cues |   The integration of Large Language Models (LLMs) and Vision-Language Models
(VLMs) opens new avenues for addressing complex challenges in multimodal
content analysis, particularly in biased news detection. This study introduces
ViLBias, a framework that leverages state of the art LLMs and VLMs to detect
linguistic and visual biases in news content, addressing the limitations of
traditional text-only approaches. Our contributions include a novel dataset
pairing textual content with accompanying visuals from diverse news sources and
a hybrid annotation framework, combining LLM-based annotations with human
review to enhance quality while reducing costs and improving scalability. We
evaluate the efficacy of LLMs and VLMs in identifying biases, revealing their
strengths in detecting subtle framing and text-visual inconsistencies.
Empirical analysis demonstrates that incorporating visual cues alongside text
enhances bias detection accuracy by 3 to 5 %, showcasing the complementary
strengths of LLMs in generative reasoning and Small Language Models (SLMs) in
classification. This study offers a comprehensive exploration of LLMs and VLMs
as tools for detecting multimodal biases in news content, highlighting both
their potential and limitations. Our research paves the way for more robust,
scalable, and nuanced approaches to media bias detection, contributing to the
broader field of natural language processing and multimodal analysis. (The data
and code will be made available for research purposes).
 | 2024-12-22T15:05:30Z | [Link](http://arxiv.org/abs/2412.17052v1) |
| Other | Modular Conversational Agents for Surveys and Interviews |   Surveys and interviews (structured, semi-structured, or unstructured) are
widely used for collecting insights on emerging or hypothetical scenarios.
Traditional human-led methods often face challenges related to cost,
scalability, and consistency. Recently, various domains have begun to explore
the use of conversational agents (chatbots) powered by large language models
(LLMs). However, as public investments and policies on infrastructure and
services often involve substantial public stakes and environmental risks, there
is a need for a rigorous, transparent, privacy-preserving, and cost-efficient
development framework tailored for such major decision-making processes. This
paper addresses this gap by introducing a modular approach and its resultant
parameterized process for designing conversational agents. We detail the system
architecture, integrating engineered prompts, specialized knowledge bases, and
customizable, goal-oriented conversational logic in the proposed approach. We
demonstrate the adaptability, generalizability, and efficacy of our modular
approach through three empirical studies: (1) travel preference surveys,
highlighting multimodal (voice, text, and image generation) capabilities; (2)
public opinion elicitation on a newly constructed, novel infrastructure
project, showcasing question customization and multilingual (English and
French) capabilities; and (3) transportation expert consultation about future
transportation systems, highlighting real-time, clarification request
capabilities for open-ended questions, resilience in handling erratic inputs,
and efficient transcript post-processing. The results show the effectiveness of
this modular approach and how it addresses key ethical, privacy, security, and
token consumption concerns, setting the stage for the next-generation surveys
and interviews.
 | 2024-12-22T15:00:16Z | [Link](http://arxiv.org/abs/2412.17049v1) |
| Other | Shaping the Safety Boundaries: Understanding and Defending Against
  Jailbreaks in Large Language Models |   Jailbreaking in Large Language Models (LLMs) is a major security concern as
it can deceive LLMs to generate harmful text. Yet, there is still insufficient
understanding of how jailbreaking works, which makes it hard to develop
effective defense strategies. We aim to shed more light into this issue: we
conduct a detailed large-scale analysis of seven different jailbreak methods
and find that these disagreements stem from insufficient observation samples.
In particular, we introduce \textit{safety boundary}, and we find that
jailbreaks shift harmful activations outside that safety boundary, where LLMs
are less sensitive to harmful information. We also find that the low and the
middle layers are critical in such shifts, while deeper layers have less
impact. Leveraging on these insights, we propose a novel defense called
\textbf{Activation Boundary Defense} (ABD), which adaptively constrains the
activations within the safety boundary. We further use Bayesian optimization to
selectively apply the defense method to the low and the middle layers. Our
experiments on several benchmarks show that ABD achieves an average DSR of over
98\% against various forms of jailbreak attacks, with less than 2\% impact on
the model's general capabilities.
 | 2024-12-22T14:18:39Z | [Link](http://arxiv.org/abs/2412.17034v1) |
| Other | MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on
  New and Tail Knowledge |   Large language models (LLMs) have demonstrated impressive capabilities in
various reasoning tasks but face significant challenges with complex,
knowledge-intensive multi-hop queries, particularly those involving new or
long-tail knowledge. Existing benchmarks often fail to fully address these
challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question
Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate
LLMs' capabilities in multi-hop reasoning across four critical dimensions:
question handling strategy, sub-question generation, retrieval-augmented
generation, and iterative or dynamic decomposition and retrieval. MINTQA
comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887
pairs for assessing long-tail knowledge, with each question equipped with
corresponding sub-questions and answers. Our systematic evaluation of 22
state-of-the-art LLMs on MINTQA reveals significant limitations in their
ability to handle complex knowledge base queries, particularly in handling new
or unpopular knowledge. Our findings highlight critical challenges and offer
insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark
is available at https://github.com/probe2/multi-hop/.
 | 2024-12-22T14:17:12Z | [Link](http://arxiv.org/abs/2412.17032v1) |
| Other | Robustness of Large Language Models Against Adversarial Attacks |   The increasing deployment of Large Language Models (LLMs) in various
applications necessitates a rigorous evaluation of their robustness against
adversarial attacks. In this paper, we present a comprehensive study on the
robustness of GPT LLM family. We employ two distinct evaluation methods to
assess their resilience. The first method introduce character-level text attack
in input prompts, testing the models on three sentiment classification
datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves
using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our
experiments reveal significant variations in the robustness of these models,
demonstrating their varying degrees of vulnerability to both character-level
and semantic-level adversarial attacks. These findings underscore the necessity
for improved adversarial training and enhanced safety mechanisms to bolster the
robustness of LLMs.
 | 2024-12-22T13:21:15Z | [Link](http://arxiv.org/abs/2412.17011v1) |
| Other | LLM-Powered User Simulator for Recommender System |   User simulators can rapidly generate a large volume of timely user behavior
data, providing a testing platform for reinforcement learning-based recommender
systems, thus accelerating their iteration and optimization. However, prevalent
user simulators generally suffer from significant limitations, including the
opacity of user preference modeling and the incapability of evaluating
simulation accuracy. In this paper, we introduce an LLM-powered user simulator
to simulate user engagement with items in an explicit manner, thereby enhancing
the efficiency and effectiveness of reinforcement learning-based recommender
systems training. Specifically, we identify the explicit logic of user
preferences, leverage LLMs to analyze item characteristics and distill user
sentiments, and design a logical model to imitate real human engagement. By
integrating a statistical model, we further enhance the reliability of the
simulation, proposing an ensemble model that synergizes logical and statistical
insights for user interaction simulations. Capitalizing on the extensive
knowledge and semantic generation capabilities of LLMs, our user simulator
faithfully emulates user behaviors and preferences, yielding high-fidelity
training data that enrich the training of recommendation algorithms. We
establish quantifying and qualifying experiments on five datasets to validate
the simulator's effectiveness and stability across various recommendation
scenarios.
 | 2024-12-22T12:00:04Z | [Link](http://arxiv.org/abs/2412.16984v1) |
| Other | Cannot or Should Not? Automatic Analysis of Refusal Composition in
  IFT/RLHF Datasets and Refusal Behavior of Black-Box LLMs |   Refusals - instances where large language models (LLMs) decline or fail to
fully execute user instructions - are crucial for both AI safety and AI
capabilities and the reduction of hallucinations in particular. These behaviors
are learned during post-training, especially in instruction fine-tuning (IFT)
and reinforcement learning from human feedback (RLHF). However, existing
taxonomies and evaluation datasets for refusals are inadequate, often focusing
solely on should-not-related (instead of cannot-related) categories, and
lacking tools for auditing refusal content in black-box LLM outputs.
  We present a comprehensive framework for classifying LLM refusals: (a) a
taxonomy of 16 refusal categories, (b) a human-annotated dataset of over 8,600
instances from publicly available IFT and RLHF datasets, (c) a synthetic
dataset with 8,000 examples for each refusal category, and (d) classifiers
trained for refusal classification.
  Our work enables precise auditing of refusal behaviors in black-box LLMs and
automatic analyses of refusal patterns in large IFT and RLHF datasets. This
facilitates the strategic adjustment of LLM refusals, contributing to the
development of more safe and reliable LLMs.
 | 2024-12-22T11:16:53Z | [Link](http://arxiv.org/abs/2412.16974v1) |
| Other | System-2 Mathematical Reasoning via Enriched Instruction Tuning |   Solving complex mathematical problems via system-2 reasoning is a natural
human skill, yet it remains a significant challenge for current large language
models (LLMs). We identify the scarcity of deliberate multi-step reasoning data
as a primary limiting factor. To this end, we introduce Enriched Instruction
Tuning (EIT), a method that enriches existing human-annotated mathematical
datasets by synergizing human and AI feedback to create fine-grained reasoning
trajectories. These datasets are then used to fine-tune open-source LLMs,
enhancing their mathematical reasoning abilities without reliance on any
symbolic verification program. Concretely, EIT is composed of two critical
steps: Enriching with Reasoning Plan (ERP) and Enriching with Reasoning Step
(ERS). The former generates a high-level plan that breaks down complex
instructions into a sequence of simpler objectives, while ERS fills in
reasoning contexts often overlooked by human annotators, creating a smoother
reasoning trajectory for LLM fine-tuning. Unlike existing CoT prompting methods
that generate reasoning chains only depending on LLM's internal knowledge, our
method leverages human-annotated initial answers as ``meta-knowledge'' to help
LLMs generate more detailed and precise reasoning processes, leading to a more
trustworthy LLM expert for complex mathematical problems. In experiments, EIT
achieves an accuracy of 84.1\% on GSM8K and 32.5\% on MATH, surpassing
state-of-the-art fine-tuning and prompting methods, and even matching the
performance of tool-augmented methods.
 | 2024-12-22T10:49:27Z | [Link](http://arxiv.org/abs/2412.16964v1) |
| Other | Aristotle: Mastering Logical Reasoning with A Logic-Complete
  Decompose-Search-Resolve Framework |   In the context of large language models (LLMs), current advanced reasoning
methods have made impressive strides in various reasoning tasks. However, when
it comes to logical reasoning tasks, major challenges remain in both efficacy
and efficiency. This is rooted in the fact that these systems fail to fully
leverage the inherent structure of logical tasks throughout the reasoning
processes such as decomposition, search, and resolution. To address this, we
propose a logic-complete reasoning framework, Aristotle, with three key
components: Logical Decomposer, Logical Search Router, and Logical Resolver. In
our framework, symbolic expressions and logical rules are comprehensively
integrated into the entire reasoning process, significantly alleviating the
bottlenecks of logical reasoning, i.e., reducing sub-task complexity,
minimizing search errors, and resolving logical contradictions. The
experimental results on several datasets demonstrate that Aristotle
consistently outperforms state-of-the-art reasoning frameworks in both accuracy
and efficiency, particularly excelling in complex logical reasoning scenarios.
We will open-source all our code at https://github.com/Aiden0526/Aristotle.
 | 2024-12-22T10:14:09Z | [Link](http://arxiv.org/abs/2412.16953v1) |
| Other | A Career Interview Dialogue System using Large Language Model-based
  Dynamic Slot Generation |   This study aims to improve the efficiency and quality of career interviews
conducted by nursing managers. To this end, we have been developing a
slot-filling dialogue system that engages in pre-interviews to collect
information on staff careers as a preparatory step before the actual
interviews. Conventional slot-filling-based interview dialogue systems have
limitations in the flexibility of information collection because the dialogue
progresses based on predefined slot sets. We therefore propose a method that
leverages large language models (LLMs) to dynamically generate new slots
according to the flow of the dialogue, achieving more natural conversations.
Furthermore, we incorporate abduction into the slot generation process to
enable more appropriate and effective slot generation. To validate the
effectiveness of the proposed method, we conducted experiments using a user
simulator. The results suggest that the proposed method using abduction is
effective in enhancing both information-collecting capabilities and the
naturalness of the dialogue.
 | 2024-12-22T09:25:02Z | [Link](http://arxiv.org/abs/2412.16943v1) |
| Other | Prompting Large Language Models with Rationale Heuristics for
  Knowledge-based Visual Question Answering |   Recently, Large Language Models (LLMs) have been used for knowledge-based
Visual Question Answering (VQA). Despite the encouraging results of previous
studies, prior methods prompt LLMs to predict answers directly, neglecting
intermediate thought processes. We argue that prior methods do not sufficiently
activate the capacities of LLMs. We propose a framework called PLRH that
Prompts LLMs with Rationale Heuristics for knowledge-based VQA. The PLRH
prompts LLMs with Chain of Thought (CoT) to generate rationale heuristics,
i.e., intermediate thought processes, and then leverages the rationale
heuristics to inspire LLMs to predict answers. Experiments show that our
approach outperforms the existing baselines by more than 2.2 and 2.1 on OK-VQA
and A-OKVQA, respectively.
 | 2024-12-22T09:14:35Z | [Link](http://arxiv.org/abs/2412.16936v1) |
| Other | Towards a Unified Paradigm: Integrating Recommendation Systems as a New
  Language in Large Models |   This paper explores the use of Large Language Models (LLMs) for sequential
recommendation, which predicts users' future interactions based on their past
behavior. We introduce a new concept, "Integrating Recommendation Systems as a
New Language in Large Models" (RSLLM), which combines the strengths of
traditional recommenders and LLMs. RSLLM uses a unique prompting method that
combines ID-based item embeddings from conventional recommendation models with
textual item features. It treats users' sequential behaviors as a distinct
language and aligns the ID embeddings with the LLM's input space using a
projector. We also propose a two-stage LLM fine-tuning framework that refines a
pretrained LLM using a combination of two contrastive losses and a language
modeling loss. The LLM is first fine-tuned using text-only prompts, followed by
target domain fine-tuning with unified prompts. This trains the model to
incorporate behavioral knowledge from the traditional sequential recommender
into the LLM. Our empirical results validate the effectiveness of our proposed
framework.
 | 2024-12-22T09:08:46Z | [Link](http://arxiv.org/abs/2412.16933v1) |
| Other | Online Preference-based Reinforcement Learning with Self-augmented
  Feedback from Large Language Model |   Preference-based reinforcement learning (PbRL) provides a powerful paradigm
to avoid meticulous reward engineering by learning rewards based on human
preferences. However, real-time human feedback is hard to obtain in online
tasks. Most work suppose there is a "scripted teacher" that utilizes privileged
predefined reward to provide preference feedback. In this paper, we propose a
RL Self-augmented Large Language Model Feedback (RL-SaLLM-F) technique that
does not rely on privileged information for online PbRL. RL-SaLLM-F leverages
the reflective and discriminative capabilities of LLM to generate
self-augmented trajectories and provide preference labels for reward learning.
First, we identify an failure issue in LLM-based preference discrimination,
specifically "query ambiguity", in online PbRL. Then LLM is employed to provide
preference labels and generate self-augmented imagined trajectories that better
achieve the task goal, thereby enhancing the quality and efficiency of
feedback. Additionally, a double-check mechanism is introduced to mitigate
randomness in the preference labels, improving the reliability of LLM feedback.
The experiment across multiple tasks in the MetaWorld benchmark demonstrates
the specific contributions of each proposed module in RL-SaLLM-F, and shows
that self-augmented LLM feedback can effectively replace the impractical
"scripted teacher" feedback. In summary, RL-SaLLM-F introduces a new direction
of feedback acquisition in online PbRL that does not rely on any online
privileged information, offering an efficient and lightweight solution with
LLM-driven feedback.
 | 2024-12-22T06:15:25Z | [Link](http://arxiv.org/abs/2412.16878v1) |
| Other | Teaching LLMs to Refine with Tools |   Large language models (LLMs) can refine their responses based on feedback,
enabling self-improvement through iterative training or test-time refinement.
However, existing methods predominantly focus on refinement within the same
reasoning format, which may lead to non-correcting behaviors. We propose CaP, a
novel approach that uses external tools to refine chain-of-thought (CoT)
responses generated by the same or other LLMs. CaP employs a two-stage training
process: supervised fine-tuning followed by preference optimization with DPO
variants. Our observations highlight the critical role of preference
optimization in enabling effective refinement. Additionally, we compare several
sampling strategies to leverage CoT and tools at inference time. Experimental
results demonstrate CaP's potential for effective cross-reasoning refinement
and efficient inference.
 | 2024-12-22T05:43:50Z | [Link](http://arxiv.org/abs/2412.16871v1) |
| Other | CoF: Coarse to Fine-Grained Image Understanding for Multi-modal Large
  Language Models |   The impressive performance of Large Language Model (LLM) has prompted
researchers to develop Multi-modal LLM (MLLM), which has shown great potential
for various multi-modal tasks. However, current MLLM often struggles to
effectively address fine-grained multi-modal challenges. We argue that this
limitation is closely linked to the models' visual grounding capabilities. The
restricted spatial awareness and perceptual acuity of visual encoders
frequently lead to interference from irrelevant background information in
images, causing the models to overlook subtle but crucial details. As a result,
achieving fine-grained regional visual comprehension becomes difficult. In this
paper, we break down multi-modal understanding into two stages, from Coarse to
Fine (CoF). In the first stage, we prompt the MLLM to locate the approximate
area of the answer. In the second stage, we further enhance the model's focus
on relevant areas within the image through visual prompt engineering, adjusting
attention weights of pertinent regions. This, in turn, improves both visual
grounding and overall performance in downstream tasks. Our experiments show
that this approach significantly boosts the performance of baseline models,
demonstrating notable generalization and effectiveness. Our CoF approach is
available online at https://github.com/Gavin001201/CoF.
 | 2024-12-22T05:42:40Z | [Link](http://arxiv.org/abs/2412.16869v1) |
| Other | GME: Improving Universal Multimodal Retrieval by Multimodal LLMs |   Universal Multimodal Retrieval (UMR) aims to enable search across various
modalities using a unified model, where queries and candidates can consist of
pure text, images, or a combination of both. Previous work has attempted to
adopt multimodal large language models (MLLMs) to realize UMR using only text
data. However, our preliminary experiments demonstrate that more diverse
multimodal training data can further unlock the potential of MLLMs. Despite its
effectiveness, the existing multimodal training data is highly imbalanced in
terms of modality, which motivates us to develop a training data synthesis
pipeline and construct a large-scale, high-quality fused-modal training
dataset. Based on the synthetic training data, we develop the General
Multimodal Embedder (GME), an MLLM-based dense retriever designed for UMR.
Furthermore, we construct a comprehensive UMR Benchmark (UMRB) to evaluate the
effectiveness of our approach. Experimental results show that our method
achieves state-of-the-art performance among existing UMR methods. Last, we
provide in-depth analyses of model scaling, training strategies, and perform
ablation studies on both the model and synthetic data.
 | 2024-12-22T04:40:24Z | [Link](http://arxiv.org/abs/2412.16855v1) |
| Other | Sim911: Towards Effective and Equitable 9-1-1 Dispatcher Training with
  an LLM-Enabled Simulation |   Emergency response services are vital for enhancing public safety by
safeguarding the environment, property, and human lives. As frontline members
of these services, 9-1-1 dispatchers have a direct impact on response times and
the overall effectiveness of emergency operations. However, traditional
dispatcher training methods, which rely on role-playing by experienced
personnel, are labor-intensive, time-consuming, and often neglect the specific
needs of underserved communities. To address these challenges, we introduce
Sim911, the first training simulation for 9-1-1 dispatchers powered by Large
Language Models (LLMs). Sim911 enhances training through three key technical
innovations: (1) knowledge construction, which utilizes archived 9-1-1 call
data to generate simulations that closely mirror real-world scenarios; (2)
context-aware controlled generation, which employs dynamic prompts and vector
bases to ensure that LLM behavior aligns with training objectives; and (3)
validation with looped correction, which filters out low-quality responses and
refines the system performance.
 | 2024-12-22T03:43:51Z | [Link](http://arxiv.org/abs/2412.16844v1) |
| Other | Online Learning from Strategic Human Feedback in LLM Fine-Tuning |   Reinforcement learning from human feedback (RLHF) has become an essential
step in fine-tuning large language models (LLMs) to align them with human
preferences. However, human labelers are selfish and have diverse preferences.
They may strategically misreport their online feedback to influence the
system's aggregation towards their own preferences. Current practice simply
averages labelers' feedback per time and fails to identify the most accurate
human labeler, leading to linear regret $\mathcal{O}(T)$ for $T$ time slots. To
our best knowledge, we are the first to study online learning mechanisms
against strategic human labelers in the LLM fine-tuning process. We formulate a
new dynamic Bayesian game and dynamically adjust human labelers' weights in the
preference aggregation, ensuring their truthful feedback and sublinear regret
$\mathcal{O}(T^{1/2})$. Simulation results demonstrate our mechanism's great
advantages over the existing benchmark schemes.
 | 2024-12-22T02:43:07Z | [Link](http://arxiv.org/abs/2412.16834v1) |
| Other | Visual Prompting with Iterative Refinement for Design Critique
  Generation |   Feedback is crucial for every design process, such as user interface (UI)
design, and automating design critiques can significantly improve the
efficiency of the design workflow. Although existing multimodal large language
models (LLMs) excel in many tasks, they often struggle with generating
high-quality design critiques -- a complex task that requires producing
detailed design comments that are visually grounded in a given design's image.
Building on recent advancements in iterative refinement of text output and
visual prompting methods, we propose an iterative visual prompting approach for
UI critique that takes an input UI screenshot and design guidelines and
generates a list of design comments, along with corresponding bounding boxes
that map each comment to a specific region in the screenshot. The entire
process is driven completely by LLMs, which iteratively refine both the text
output and bounding boxes using few-shot samples tailored for each step. We
evaluated our approach using Gemini-1.5-pro and GPT-4o, and found that human
experts generally preferred the design critiques generated by our pipeline over
those by the baseline, with the pipeline reducing the gap from human
performance by 50% for one rating metric. To assess the generalizability of our
approach to other multimodal tasks, we applied our pipeline to open-vocabulary
object and attribute detection, and experiments showed that our method also
outperformed the baseline.
 | 2024-12-22T02:35:57Z | [Link](http://arxiv.org/abs/2412.16829v1) |
| Other | An Exploration of Pattern Mining with ChatGPT |   This paper takes an exploratory approach to examine the use of ChatGPT for
pattern mining. It proposes an eight-step collaborative process that combines
human insight with AI capabilities to extract patterns from known uses. The
paper offers a practical demonstration of this process by creating a pattern
language for integrating Large Language Models (LLMs) with data sources and
tools. LLMs, such as ChatGPT, are a new class of AI models that have been
trained on large amounts of text, and can create new content, including text,
images, or video. The paper also argues for adding affordances of the
underlying components as a new element of pattern descriptions. The primary
audience of the paper includes pattern writers interested in pattern mining
using LLMs.
 | 2024-12-22T01:27:12Z | [Link](http://arxiv.org/abs/2412.16814v1) |
| Technology | Large Language Model Safety: A Holistic Survey |   The rapid development and deployment of large language models (LLMs) have
introduced a new frontier in artificial intelligence, marked by unprecedented
capabilities in natural language understanding and generation. However, the
increasing integration of these models into critical applications raises
substantial safety concerns, necessitating a thorough examination of their
potential risks and associated mitigation strategies.
  This survey provides a comprehensive overview of the current landscape of LLM
safety, covering four major categories: value misalignment, robustness to
adversarial attacks, misuse, and autonomous AI risks. In addition to the
comprehensive review of the mitigation methodologies and evaluation resources
on these four aspects, we further explore four topics related to LLM safety:
the safety implications of LLM agents, the role of interpretability in
enhancing LLM safety, the technology roadmaps proposed and abided by a list of
AI companies and institutes for LLM safety, and AI governance aimed at LLM
safety with discussions on international cooperation, policy proposals, and
prospective regulatory directions.
  Our findings underscore the necessity for a proactive, multifaceted approach
to LLM safety, emphasizing the integration of technical solutions, ethical
considerations, and robust governance frameworks. This survey is intended to
serve as a foundational resource for academy researchers, industry
practitioners, and policymakers, offering insights into the challenges and
opportunities associated with the safe integration of LLMs into society.
Ultimately, it seeks to contribute to the safe and beneficial development of
LLMs, aligning with the overarching goal of harnessing AI for societal
advancement and well-being. A curated list of related papers has been publicly
available at https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.
 | 2024-12-23T16:11:27Z | [Link](http://arxiv.org/abs/2412.17686v1) |
| Technology | Enhancing Supply Chain Transparency in Emerging Economies Using Online
  Contents and LLMs |   In the current global economy, supply chain transparency plays a pivotal role
in ensuring this security by enabling companies to monitor supplier performance
and fostering accountability and responsibility. Despite the advancements in
supply chain relationship datasets like Bloomberg and FactSet, supply chain
transparency remains a significant challenge in emerging economies due to
issues such as information asymmetry and institutional gaps in regulation. This
study proposes a novel approach to enhance supply chain transparency in
emerging economies by leveraging online content and large language models
(LLMs). We develop a Supply Chain Knowledge Graph Mining System that integrates
advanced LLMs with web crawler technology to automatically collect and analyze
supply chain information. The system's effectiveness is validated through a
case study focusing on the semiconductor supply chain, a domain that has
recently gained significant attention due to supply chain risks. Our results
demonstrate that the proposed system provides greater applicability for
emerging economies, such as mainland China, complementing the data gaps in
existing datasets. However, challenges including the accurate estimation of
monetary and material flows, the handling of time series data, synonyms
disambiguation, and mitigating biases from online contents still remains.
Future research should focus on addressing these issues to further enhance the
system's capabilities and broaden its application to other emerging economies
and industries.
 | 2024-12-22T08:46:16Z | [Link](http://arxiv.org/abs/2412.16922v1) |
| Health | Detecting anxiety and depression in dialogues: a multi-label and
  explainable approach |   Anxiety and depression are the most common mental health issues worldwide,
affecting a non-negligible part of the population. Accordingly, stakeholders,
including governments' health systems, are developing new strategies to promote
early detection and prevention from a holistic perspective (i.e., addressing
several disorders simultaneously). In this work, an entirely novel system for
the multi-label classification of anxiety and depression is proposed. The input
data consists of dialogues from user interactions with an assistant chatbot.
Another relevant contribution lies in using Large Language Models (LLMs) for
feature extraction, provided the complexity and variability of language. The
combination of LLMs, given their high capability for language understanding,
and Machine Learning (ML) models, provided their contextual knowledge about the
classification problem thanks to the labeled data, constitute a promising
approach towards mental health assessment. To promote the solution's
trustworthiness, reliability, and accountability, explainability descriptions
of the model's decision are provided in a graphical dashboard. Experimental
results on a real dataset attain 90 % accuracy, improving those in the prior
literature. The ultimate objective is to contribute in an accessible and
scalable way before formal treatment occurs in the healthcare systems.
 | 2024-12-23T15:29:46Z | [Link](http://arxiv.org/abs/2412.17651v1) |
| Health | Emerging Security Challenges of Large Language Models |   Large language models (LLMs) have achieved record adoption in a short period
of time across many different sectors including high importance areas such as
education [4] and healthcare [23]. LLMs are open-ended models trained on
diverse data without being tailored for specific downstream tasks, enabling
broad applicability across various domains. They are commonly used for text
generation, but also widely used to assist with code generation [3], and even
analysis of security information, as Microsoft Security Copilot demonstrates
[18]. Traditional Machine Learning (ML) models are vulnerable to adversarial
attacks [9]. So the concerns on the potential security implications of such
wide scale adoption of LLMs have led to the creation of this working group on
the security of LLMs. During the Dagstuhl seminar on "Network Attack Detection
and Defense - AI-Powered Threats and Responses", the working group discussions
focused on the vulnerability of LLMs to adversarial attacks, rather than their
potential use in generating malware or enabling cyberattacks. Although we note
the potential threat represented by the latter, the role of the LLMs in such
uses is mostly as an accelerator for development, similar to what it is in
benign use. To make the analysis more specific, the working group employed
ChatGPT as a concrete example of an LLM and addressed the following points,
which also form the structure of this report: 1. How do LLMs differ in
vulnerabilities from traditional ML models? 2. What are the attack objectives
in LLMs? 3. How complex it is to assess the risks posed by the vulnerabilities
of LLMs? 4. What is the supply chain in LLMs, how data flow in and out of
systems and what are the security implications? We conclude with an overview of
open challenges and outlook.
 | 2024-12-23T14:36:37Z | [Link](http://arxiv.org/abs/2412.17614v1) |
| Health | PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality
  and Mental Health |   Artificial intelligence-based language generators are now a part of most
people's lives. However, by default, they tend to generate "average" language
without reflecting the ways in which people differ. Here, we propose a
lightweight modification to the standard language model transformer
architecture - "PsychAdapter" - that uses empirically derived trait-language
patterns to generate natural language for specified personality, demographic,
and mental health characteristics (with or without prompting). We applied
PsychAdapters to modify OpenAI's GPT-2, Google's Gemma, and Meta's Llama 3 and
found generated text to reflect the desired traits. For example, expert raters
evaluated PsychAdapter's generated text output and found it matched intended
trait levels with 87.3% average accuracy for Big Five personalities, and 96.7%
for depression and life satisfaction. PsychAdapter is a novel method to
introduce psychological behavior patterns into language models at the
foundation level, independent of prompting, by influencing every transformer
layer. This approach can create chatbots with specific personality profiles,
clinical training tools that mirror language associated with psychological
conditionals, and machine translations that match an authors reading or
education level without taking up LLM context windows. PsychAdapter also allows
for the exploration psychological constructs through natural language
expression, extending the natural language processing toolkit to study human
psychology.
 | 2024-12-22T06:22:40Z | [Link](http://arxiv.org/abs/2412.16882v1) |
| Health | KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge
  Graph Enhancement for Medical Diagnosis |   Integrating Large Language Models (LLMs) in healthcare diagnosis demands
systematic frameworks that can handle complex medical scenarios while
maintaining specialized expertise. We present KG4Diagnosis, a novel
hierarchical multi-agent framework that combines LLMs with automated knowledge
graph construction, encompassing 362 common diseases across medical
specialties. Our framework mirrors real-world medical systems through a
two-tier architecture: a general practitioner (GP) agent for initial assessment
and triage, coordinating with specialized agents for in-depth diagnosis in
specific domains. The core innovation lies in our end-to-end knowledge graph
generation methodology, incorporating: (1) semantic-driven entity and relation
extraction optimized for medical terminology, (2) multi-dimensional decision
relationship reconstruction from unstructured medical texts, and (3)
human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an
extensible foundation for specialized medical diagnosis systems, with
capabilities to incorporate new diseases and medical knowledge. The framework's
modular design enables seamless integration of domain-specific enhancements,
making it valuable for developing targeted medical diagnosis systems. We
provide architectural guidelines and protocols to facilitate adoption across
medical contexts.
 | 2024-12-22T02:40:59Z | [Link](http://arxiv.org/abs/2412.16833v1) |
| Education | Is ChatGPT Massively Used by Students Nowadays? A Survey on the Use of
  Large Language Models such as ChatGPT in Educational Settings |   The rapid adoption of Generative AI (GenAI) based on Large Language Models
(LLMs) such as ChatGPT has recently and profoundly impacted education, offering
transformative opportunities while raising significant concerns. In this study
we present the results of a survey that investigates how 395 students aged 13
to 25 years old in France and Italy integrate LLMs into their educational
routines.
  Key findings include the widespread use of these tools across all age groups
and disciplines, with older students and male students demonstrating higher
usage frequencies, particularly in scientific contexts. The results also show
gender disparities, raising concerns about an emerging AI literacy and
technological gender gap. Additionally, while most students utilise LLMs
constructively, the lack of systematic proofreading and critical evaluation
among younger users suggests potential risks to cognitive skills development,
including critical thinking and foundational knowledge. The survey results
underscore the need for educational institutions to adapt their curricula to
integrate AI tools effectively, promoting ethical use, critical thinking, and
awareness of AI limitations and environmental costs. This paper provides
actionable recommendations for fostering equitable and effective cohabitation
of LLMs and education while addressing emerging challenges.
 | 2024-12-23T11:29:44Z | [Link](http://arxiv.org/abs/2412.17486v1) |
| Education | Measuring Contextual Informativeness in Child-Directed Text |   To address an important gap in creating children's stories for vocabulary
enrichment, we investigate the automatic evaluation of how well stories convey
the semantics of target vocabulary words, a task with substantial implications
for generating educational content. We motivate this task, which we call
measuring contextual informativeness in children's stories, and provide a
formal task definition as well as a dataset for the task. We further propose a
method for automating the task using a large language model (LLM). Our
experiments show that our approach reaches a Spearman correlation of 0.4983
with human judgments of informativeness, while the strongest baseline only
obtains a correlation of 0.3534. An additional analysis shows that the
LLM-based approach is able to generalize to measuring contextual
informativeness in adult-directed text, on which it also outperforms all
baselines.
 | 2024-12-23T09:45:03Z | [Link](http://arxiv.org/abs/2412.17427v1) |
| Education | Ask-Before-Detection: Identifying and Mitigating Conformity Bias in
  LLM-Powered Error Detector for Math Word Problem Solutions |   The rise of large language models (LLMs) offers new opportunities for
automatic error detection in education, particularly for math word problems
(MWPs). While prior studies demonstrate the promise of LLMs as error detectors,
they overlook the presence of multiple valid solutions for a single MWP. Our
preliminary analysis reveals a significant performance gap between conventional
and alternative solutions in MWPs, a phenomenon we term conformity bias in this
work. To mitigate this bias, we introduce the Ask-Before-Detect (AskBD)
framework, which generates adaptive reference solutions using LLMs to enhance
error detection. Experiments on 200 examples of GSM8K show that AskBD
effectively mitigates bias and improves performance, especially when combined
with reasoning-enhancing techniques like chain-of-thought prompting.
 | 2024-12-22T03:08:36Z | [Link](http://arxiv.org/abs/2412.16838v1) |