# Recent LLM Applications

| Domain      | Title | Summary | Published | Link |
|------------|-------|---------|-----------|------|
| Other | Long-Form Speech Generation with Spoken Language Models |   We consider the generative modeling of speech over multiple minutes, a
requirement for long-form multimedia generation and audio-native voice
assistants. However, current spoken language models struggle to generate
plausible speech past tens of seconds, from high temporal resolution of speech
tokens causing loss of coherence, to architectural issues with long-sequence
training or extrapolation, to memory costs at inference time. With these
considerations we propose SpeechSSM, the first speech language model to learn
from and sample long-form spoken audio (e.g., 16 minutes of read or
extemporaneous speech) in a single decoding session without text intermediates,
based on recent advances in linear-time sequence modeling. Furthermore, to
address growing challenges in spoken language evaluation, especially in this
new long-form setting, we propose: new embedding-based and LLM-judged metrics;
quality measurements over length and time; and a new benchmark for long-form
speech processing and generation, LibriSpeech-Long. Speech samples and the
dataset are released at
https://google.github.io/tacotron/publications/speechssm/
 | 2024-12-24T18:56:46Z | [Link](http://arxiv.org/abs/2412.18603v1) |
| Other | A Paragraph is All It Takes: Rich Robot Behaviors from Interacting,
  Trusted LLMs |   Large Language Models (LLMs) are compact representations of all public
knowledge of our physical environment and animal and human behaviors. The
application of LLMs to robotics may offer a path to highly capable robots that
perform well across most human tasks with limited or even zero tuning. Aside
from increasingly sophisticated reasoning and task planning, networks of
(suitably designed) LLMs offer ease of upgrading capabilities and allow humans
to directly observe the robot's thinking. Here we explore the advantages,
limitations, and particularities of using LLMs to control physical robots. The
basic system consists of four LLMs communicating via a human language data bus
implemented via web sockets and ROS2 message passing. Surprisingly, rich robot
behaviors and good performance across different tasks could be achieved despite
the robot's data fusion cycle running at only 1Hz and the central data bus
running at the extremely limited rates of the human brain, of around 40 bits/s.
The use of natural language for inter-LLM communication allowed the robot's
reasoning and decision making to be directly observed by humans and made it
trivial to bias the system's behavior with sets of rules written in plain
English. These rules were immutably written into Ethereum, a global, public,
and censorship resistant Turing-complete computer. We suggest that by using
natural language as the data bus among interacting AIs, and immutable public
ledgers to store behavior constraints, it is possible to build robots that
combine unexpectedly rich performance, upgradability, and durable alignment
with humans.
 | 2024-12-24T18:41:15Z | [Link](http://arxiv.org/abs/2412.18588v1) |
| Other | How Well Do LLMs Generate Code for Different Application Domains?
  Benchmark and Evaluation |   Recently, an increasing number of AI-driven programming assistants powered by
code LLMs have been integrated into various real-world software development
environments, significantly boosting developer productivity. However, existing
code generation benchmarks primarily focus on general-purpose scenarios,
leaving the code generation performance of LLMs for specific application
domains largely unknown. In this paper, we introduce a new benchmark,
MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming
tasks, covering 12 popular software development domains and 15 programming
languages. Specifically, we perform in-depth research to identify these 12
application domains. Given that each domain may involve multiple technical
frameworks, and that different frameworks present distinct challenges in the
coding process, we categorize the commonly used frameworks and platforms within
each domain. We then sample programming problems from GitHub repositories
related to these subdomains. To ensure the quality of the tasks and mitigate
data leakage issues, we invite annotators to rewrite the docstrings for each
task in MultiCodeBench. Additionally, we build a static analysis-based
dependency parsing tool to extract the dependencies in the ground truth for
each task, enabling deeper performance analysis. Through extensive experiments
on MultiCodeBench with eleven representative mainstream LLMs, we reveal the
code generation performance of the LLMs across different application domains,
providing practical insights for developers in downstream fields when selecting
LLMs. Furthermore, we analyze the reasons behind the models' failures in
completing software application development tasks, offering guidance for model
developers to enhance domain-specific code generation capabilities.
 | 2024-12-24T17:56:08Z | [Link](http://arxiv.org/abs/2412.18573v1) |
| Other | Zero-resource Speech Translation and Recognition with LLMs |   Despite recent advancements in speech processing, zero-resource speech
translation (ST) and automatic speech recognition (ASR) remain challenging
problems. In this work, we propose to leverage a multilingual Large Language
Model (LLM) to perform ST and ASR in languages for which the model has never
seen paired audio-text data. We achieve this by using a pre-trained
multilingual speech encoder, a multilingual LLM, and a lightweight adaptation
module that maps the audio representations to the token embedding space of the
LLM. We perform several experiments both in ST and ASR to understand how to
best train the model and what data has the most impact on performance in
previously unseen languages. In ST, our best model is capable to achieve BLEU
scores over 23 in CoVoST2 for two previously unseen languages, while in ASR, we
achieve WERs of up to 28.2\%. We finally show that the performance of our
system is bounded by the ability of the LLM to output text in the desired
language.
 | 2024-12-24T17:37:11Z | [Link](http://arxiv.org/abs/2412.18566v1) |
| Other | Distilling Fine-grained Sentiment Understanding from Large Language
  Models |   Fine-grained sentiment analysis (FSA) aims to extract and summarize user
opinions from vast opinionated text. Recent studies demonstrate that large
language models (LLMs) possess exceptional sentiment understanding
capabilities. However, directly deploying LLMs for FSA applications incurs high
inference costs. Therefore, this paper investigates the distillation of
fine-grained sentiment understanding from LLMs into small language models
(SLMs). We prompt LLMs to examine and interpret the sentiments of given reviews
and then utilize the generated content to pretrain SLMs. Additionally, we
develop a comprehensive FSA benchmark to evaluate both SLMs and LLMs. Extensive
experiments on this benchmark reveal that: (1) distillation significantly
enhances the performance of SLMs in FSA tasks, achieving a 6.00\% improvement
in $F_1$-score, and the distilled model can outperform Llama-2-7b with only
220M parameters; (2) distillation equips SLMs with excellent zero-shot
sentiment classification capabilities, enabling them to match or even exceed
their teacher models. These results suggest that distillation from LLMs is a
highly promising direction for FSA. We will release our code, data, and
pretrained model weights at
\url{https://github.com/HITSZ-HLT/FSA-Distillation}.
 | 2024-12-24T17:05:26Z | [Link](http://arxiv.org/abs/2412.18552v1) |
| Other | Libra-Leaderboard: Towards Responsible AI through a Balanced Leaderboard
  of Safety and Capability |   To address this gap, we introduce Libra-Leaderboard, a comprehensive
framework designed to rank LLMs through a balanced evaluation of performance
and safety. Combining a dynamic leaderboard with an interactive LLM arena,
Libra-Leaderboard encourages the joint optimization of capability and safety.
Unlike traditional approaches that average performance and safety metrics,
Libra-Leaderboard uses a distance-to-optimal-score method to calculate the
overall rankings. This approach incentivizes models to achieve a balance rather
than excelling in one dimension at the expense of some other ones. In the first
release, Libra-Leaderboard evaluates 26 mainstream LLMs from 14 leading
organizations, identifying critical safety challenges even in state-of-the-art
models.
 | 2024-12-24T17:03:44Z | [Link](http://arxiv.org/abs/2412.18551v1) |
| Other | Token-Budget-Aware LLM Reasoning |   Reasoning is critical for large language models (LLMs) to excel in a wide
range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM
performance by decomposing problems into intermediate steps, they also incur
significant overhead in token usage, leading to increased costs. We find that
the reasoning process of current LLMs is unnecessarily lengthy and it can be
compressed by including a reasonable token budget in the prompt, but the choice
of token budget plays a crucial role in the actual compression effectiveness.
We then propose a token-budget-aware LLM reasoning framework, which dynamically
estimates token budgets for different problems based on reasoning complexity
and uses the estimated token budgets to guide the reasoning process.
Experiments show that our method effectively reduces token costs in CoT
reasoning with only a slight performance reduction, offering a practical
solution to balance efficiency and accuracy in LLM reasoning. Code:
https://github.com/GeniusHTX/TALE.
 | 2024-12-24T16:55:45Z | [Link](http://arxiv.org/abs/2412.18547v1) |
| Other | Consistency Checks for Language Model Forecasters |   Forecasting is a task that is difficult to evaluate: the ground truth can
only be known in the future. Recent work showing LLM forecasters rapidly
approaching human-level performance begs the question: how can we benchmark and
evaluate these forecasters instantaneously? Following the consistency check
framework, we measure the performance of forecasters in terms of the
consistency of their predictions on different logically-related questions. We
propose a new, general consistency metric based on arbitrage: for example, if a
forecasting AI illogically predicts that both the Democratic and Republican
parties have 60% probability of winning the 2024 US presidential election, an
arbitrageur can trade against the forecaster's predictions and make a profit.
We build an automated evaluation system that generates a set of base questions,
instantiates consistency checks from these questions, elicits the predictions
of the forecaster, and measures the consistency of the predictions. We then
build a standard, proper-scoring-rule forecasting benchmark, and show that our
(instantaneous) consistency metrics correlate with LLM forecasters' ground
truth Brier scores (which are only known in the future). We also release a
consistency benchmark that resolves in 2028, providing a long-term evaluation
tool for forecasting.
 | 2024-12-24T16:51:35Z | [Link](http://arxiv.org/abs/2412.18544v1) |
| Other | Harnessing Large Language Models for Knowledge Graph Question Answering
  via Adaptive Multi-Aspect Retrieval-Augmentation |   Large Language Models (LLMs) demonstrate remarkable capabilities, yet
struggle with hallucination and outdated knowledge when tasked with complex
knowledge reasoning, resulting in factually incorrect outputs. Previous studies
have attempted to mitigate it by retrieving factual knowledge from large-scale
knowledge graphs (KGs) to assist LLMs in logical reasoning and prediction of
answers. However, this kind of approach often introduces noise and irrelevant
data, especially in situations with extensive context from multiple knowledge
aspects. In this way, LLM attention can be potentially mislead from question
and relevant information. In our study, we introduce an Adaptive Multi-Aspect
Retrieval-augmented over KGs (Amar) framework. This method retrieves knowledge
including entities, relations, and subgraphs, and converts each piece of
retrieved text into prompt embeddings. The Amar framework comprises two key
sub-components: 1) a self-alignment module that aligns commonalities among
entities, relations, and subgraphs to enhance retrieved text, thereby reducing
noise interference; 2) a relevance gating module that employs a soft gate to
learn the relevance score between question and multi-aspect retrieved data, to
determine which information should be used to enhance LLMs' output, or even
filtered altogether. Our method has achieved state-of-the-art performance on
two common datasets, WebQSP and CWQ, showing a 1.9\% improvement in accuracy
over its best competitor and a 6.6\% improvement in logical form generation
over a method that directly uses retrieved text as context prompts. These
results demonstrate the effectiveness of Amar in improving the reasoning of
LLMs.
 | 2024-12-24T16:38:04Z | [Link](http://arxiv.org/abs/2412.18537v1) |
| Other | Automated Code Review In Practice |   Code review is a widespread practice to improve software quality and transfer
knowledge. It is often seen as time-consuming due to the need for manual effort
and potential delays. Several AI-assisted tools, such as Qodo, GitHub Copilot,
and Coderabbit, provide automated reviews using large language models (LLMs).
The effects of such tools in the industry are yet to be examined.
  This study examines the impact of LLM-based automated code review tools in an
industrial setting. The study was conducted within a software development
environment that adopted an AI-assisted review tool (based on open-source Qodo
PR Agent). Around 238 practitioners across ten projects had access to the tool.
We focused on three projects with 4,335 pull requests, 1,568 of which underwent
automated reviews. Data collection comprised three sources: (1) a quantitative
analysis of pull request data, including comment labels indicating whether
developers acted on the automated comments, (2) surveys sent to developers
regarding their experience with reviews on individual pull requests, and (3) a
broader survey of 22 practitioners capturing their general opinions on
automated reviews.
  73.8% of automated comments were resolved. However, the average pull request
closure duration increased from five hours 52 minutes to eight hours 20
minutes, with varying trends across projects. Most practitioners reported a
minor improvement in code quality due to automated reviews.
  The LLM-based tool proved useful in software development, enhancing bug
detection, increasing awareness of code quality, and promoting best practices.
However, it also led to longer pull request closure times and introduced
drawbacks like faulty reviews, unnecessary corrections, and irrelevant
comments.
 | 2024-12-24T16:24:45Z | [Link](http://arxiv.org/abs/2412.18531v1) |
| Other | Large Language Model guided Deep Reinforcement Learning for Decision
  Making in Autonomous Driving |   Deep reinforcement learning (DRL) shows promising potential for autonomous
driving decision-making. However, DRL demands extensive computational resources
to achieve a qualified policy in complex driving scenarios due to its low
learning efficiency. Moreover, leveraging expert guidance from human to enhance
DRL performance incurs prohibitively high labor costs, which limits its
practical application. In this study, we propose a novel large language model
(LLM) guided deep reinforcement learning (LGDRL) framework for addressing the
decision-making problem of autonomous vehicles. Within this framework, an
LLM-based driving expert is integrated into the DRL to provide intelligent
guidance for the learning process of DRL. Subsequently, in order to efficiently
utilize the guidance of the LLM expert to enhance the performance of DRL
decision-making policies, the learning and interaction process of DRL is
enhanced through an innovative expert policy constrained algorithm and a novel
LLM-intervened interaction mechanism. Experimental results demonstrate that our
method not only achieves superior driving performance with a 90\% task success
rate but also significantly improves the learning efficiency and expert
guidance utilization efficiency compared to state-of-the-art baseline
algorithms. Moreover, the proposed method enables the DRL agent to maintain
consistent and reliable performance in the absence of LLM expert guidance. The
code and supplementary videos are available at
https://bitmobility.github.io/LGDRL/.
 | 2024-12-24T15:50:10Z | [Link](http://arxiv.org/abs/2412.18511v1) |
| Other | Think or Remember? Detecting and Directing LLMs Towards Memorization or
  Generalization |   In this paper, we explore the foundational mechanisms of memorization and
generalization in Large Language Models (LLMs), inspired by the functional
specialization observed in the human brain. Our investigation serves as a case
study leveraging specially designed datasets and experimental-scale LLMs to lay
the groundwork for understanding these behaviors. Specifically, we aim to first
enable LLMs to exhibit both memorization and generalization by training with
the designed dataset, then (a) examine whether LLMs exhibit neuron-level
spatial differentiation for memorization and generalization, (b) predict these
behaviors using model internal representations, and (c) steer the behaviors
through inference-time interventions. Our findings reveal that neuron-wise
differentiation of memorization and generalization is observable in LLMs, and
targeted interventions can successfully direct their behavior.
 | 2024-12-24T15:28:56Z | [Link](http://arxiv.org/abs/2412.18497v1) |
| Other | 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D
  Scene Understanding |   A 3D scene graph represents a compact scene model, storing information about
the objects and the semantic relationships between them, making its use
promising for robotic tasks. When interacting with a user, an embodied
intelligent agent should be capable of responding to various queries about the
scene formulated in natural language. Large Language Models (LLMs) are
beneficial solutions for user-robot interaction due to their natural language
understanding and reasoning abilities. Recent methods for creating learnable
representations of 3D scenes have demonstrated the potential to improve the
quality of LLMs responses by adapting to the 3D world. However, the existing
methods do not explicitly utilize information about the semantic relationships
between objects, limiting themselves to information about their coordinates. In
this work, we propose a method 3DGraphLLM for constructing a learnable
representation of a 3D scene graph. The learnable representation is used as
input for LLMs to perform 3D vision-language tasks. In our experiments on
popular ScanRefer, RIORefer, Multi3DRefer, ScanQA, Sqa3D, and Scan2cap
datasets, we demonstrate the advantage of this approach over baseline methods
that do not use information about the semantic relationships between objects.
The code is publicly available at
https://github.com/CognitiveAISystems/3DGraphLLM.
 | 2024-12-24T14:21:58Z | [Link](http://arxiv.org/abs/2412.18450v1) |
| Other | Is Large Language Model Good at Triple Set Prediction? An Empirical
  Study |   The core of the Knowledge Graph Completion (KGC) task is to predict and
complete the missing relations or nodes in a KG. Common KGC tasks are mostly
about inferring unknown elements with one or two elements being known in a
triple. In comparison, the Triple Set Prediction (TSP) task is a more realistic
knowledge graph completion task. It aims to predict all elements of unknown
triples based on the information from known triples. In recent years, large
language models (LLMs) have exhibited significant advancements in language
comprehension, demonstrating considerable potential for KGC tasks. However, the
potential of LLM on the TSP task has not yet to be investigated. Thus in this
paper we proposed a new framework to explore the strengths and limitations of
LLM in the TSP task. Specifically, the framework consists of LLM-based rule
mining and LLM-based triple set prediction. The relation list of KG embedded
within rich semantic information is first leveraged to prompt LLM in the
generation of rules. This process is both efficient and independent of
statistical information, making it easier to mine effective and realistic
rules. For each subgraph, the specified rule is applied in conjunction with the
relevant triples within that subgraph to guide the LLM in predicting the
missing triples. Subsequently, the predictions from all subgraphs are
consolidated to derive the complete set of predicted triples on KG. Finally,
the method is evaluated on the relatively complete CFamily dataset. The
experimental results indicate that when LLMs are required to adhere to a large
amount of factual knowledge to predict missing triples, significant
hallucinations occurs, leading to a noticeable decline in performance. To
further explore the causes of this phenomenon, this paper presents a
comprehensive analysis supported by a detailed case study.
 | 2024-12-24T14:03:07Z | [Link](http://arxiv.org/abs/2412.18443v1) |
| Other | Explainable Multi-Modal Data Exploration in Natural Language via LLM
  Agent |   International enterprises, organizations, or hospitals collect large amounts
of multi-modal data stored in databases, text documents, images, and videos.
While there has been recent progress in the separate fields of multi-modal data
exploration as well as in database systems that automatically translate natural
language questions to database query languages, the research challenge of
querying database systems combined with other unstructured modalities such as
images in natural language is widely unexplored.
  In this paper, we propose XMODE - a system that enables explainable,
multi-modal data exploration in natural language. Our approach is based on the
following research contributions: (1) Our system is inspired by a real-world
use case that enables users to explore multi-modal information systems. (2)
XMODE leverages a LLM-based agentic AI framework to decompose a natural
language question into subtasks such as text-to-SQL generation and image
analysis. (3) Experimental results on multi-modal datasets over relational data
and images demonstrate that our system outperforms state-of-the-art multi-modal
exploration systems, excelling not only in accuracy but also in various
performance metrics such as query latency, API costs, planning efficiency, and
explanation quality, thanks to the more effective utilization of the reasoning
capabilities of LLMs.
 | 2024-12-24T13:42:44Z | [Link](http://arxiv.org/abs/2412.18428v1) |
| Other | Muse: A Multimodal Conversational Recommendation Dataset with
  Scenario-Grounded User Profiles |   Current conversational recommendation systems focus predominantly on text.
However, real-world recommendation settings are generally multimodal, causing a
significant gap between existing research and practical applications. To
address this issue, we propose Muse, the first multimodal conversational
recommendation dataset. Muse comprises 83,148 utterances from 7,000
conversations centered around the Clothing domain. Each conversation contains
comprehensive multimodal interactions, rich elements, and natural dialogues.
Data in Muse are automatically synthesized by a multi-agent framework powered
by multimodal large language models (MLLMs). It innovatively derives user
profiles from real-world scenarios rather than depending on manual design and
history data for better scalability, and then it fulfills conversation
simulation and optimization. Both human and LLM evaluations demonstrate the
high quality of conversations in Muse. Additionally, fine-tuning experiments on
three MLLMs demonstrate Muse's learnable patterns for recommendations and
responses, confirming its value for multimodal conversational recommendation.
Our dataset and codes are available at
\url{https://anonymous.4open.science/r/Muse-0086}.
 | 2024-12-24T13:08:34Z | [Link](http://arxiv.org/abs/2412.18416v1) |
| Other | Multilingual Mathematical Reasoning: Advancing Open-Source LLMs in Hindi
  and English |   Large Language Models (LLMs) excel in linguistic tasks but struggle with
mathematical reasoning, particularly in non English languages like Hindi. This
research aims to enhance the mathematical reasoning skills of smaller, resource
efficient open-source LLMs in both Hindi and English. We evaluate models like
OpenHathi 7B, LLaMA-2 7B, WizardMath 7B, Mistral 7B, LLeMMa 7B, MAmmoTH 7B,
Gemini Pro, and GPT-4 using zero-shot, few-shot chain-of-thought (CoT) methods,
and supervised fine-tuning. Our approach incorporates curriculum learning,
progressively training models on increasingly difficult problems, a novel
Decomposition Strategy to simplify complex arithmetic operations, and a
Structured Solution Design that divides solutions into phases. Our experiments
result in notable performance enhancements. WizardMath 7B exceeds Gemini's
accuracy on English datasets by +6% and matches Gemini's performance on Hindi
datasets. Adopting a bilingual approach that combines English and Hindi samples
achieves results comparable to individual language models, demonstrating the
capability to learn mathematical reasoning in both languages. This research
highlights the potential for improving mathematical reasoning in open-source
LLMs.
 | 2024-12-24T13:07:29Z | [Link](http://arxiv.org/abs/2412.18415v1) |
| Other | A Statistical Framework for Ranking LLM-Based Chatbots |   Large language models (LLMs) have transformed natural language processing,
with frameworks like Chatbot Arena providing pioneering platforms for
evaluating these models. By facilitating millions of pairwise comparisons based
on human judgments, Chatbot Arena has become a cornerstone in LLM evaluation,
offering rich datasets for ranking models in open-ended conversational tasks.
Building upon this foundation, we propose a statistical framework that
incorporates key advancements to address specific challenges in pairwise
comparison analysis. First, we introduce a factored tie model that enhances the
ability to handle ties -- an integral aspect of human-judged comparisons --
significantly improving the model's fit to observed data. Second, we extend the
framework to model covariance between competitors, enabling deeper insights
into performance relationships and facilitating intuitive groupings into
performance tiers. Third, we resolve optimization challenges arising from
parameter non-uniqueness by introducing novel constraints, ensuring stable and
interpretable parameter estimation. Through rigorous evaluation and extensive
experimentation, our framework demonstrates substantial improvements over
existing methods in modeling pairwise comparison data. To support
reproducibility and practical adoption, we release leaderbot, an open-source
Python package implementing our models and analyses.
 | 2024-12-24T12:54:19Z | [Link](http://arxiv.org/abs/2412.18407v1) |
| Other | RDPM: Solve Diffusion Probabilistic Models via Recurrent Token
  Prediction |   Diffusion Probabilistic Models (DPMs) have emerged as the de facto approach
for high-fidelity image synthesis, operating diffusion processes on continuous
VAE latent, which significantly differ from the text generation methods
employed by Large Language Models (LLMs). In this paper, we introduce a novel
generative framework, the Recurrent Diffusion Probabilistic Model (RDPM), which
enhances the diffusion process through a recurrent token prediction mechanism,
thereby pioneering the field of Discrete Diffusion. By progressively
introducing Gaussian noise into the latent representations of images and
encoding them into vector-quantized tokens in a recurrent manner, RDPM
facilitates a unique diffusion process on discrete-value domains. This process
iteratively predicts the token codes for subsequent timesteps, transforming the
initial standard Gaussian noise into the source data distribution, aligning
with GPT-style models in terms of the loss function. RDPM demonstrates superior
performance while benefiting from the speed advantage of requiring only a few
inference steps. This model not only leverages the diffusion process to ensure
high-quality generation but also converts continuous signals into a series of
high-fidelity discrete tokens, thereby maintaining a unified optimization
strategy with other discrete tokens, such as text. We anticipate that this work
will contribute to the development of a unified model for multimodal
generation, specifically by integrating continuous signal domains such as
images, videos, and audio with text. We will release the code and model weights
to the open-source community.
 | 2024-12-24T12:28:19Z | [Link](http://arxiv.org/abs/2412.18390v1) |
| Other | ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with
  LLM-based Chatbots |   The rise of LLMs has deflected a growing portion of human-computer
interactions towards LLM-based chatbots. The remarkable abilities of these
models allow users to interact using long, diverse natural language text
covering a wide range of topics and styles. Phrasing these messages is a time
and effort consuming task, calling for an autocomplete solution to assist
users. We introduce the task of chatbot interaction autocomplete. We present
ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework
for LLM-based chatbot interactions. The framework includes a formal definition
of the task, coupled with suitable datasets and metrics. We use the framework
to evaluate After formally defining the task along with suitable datasets and
metrics, we test 9 models on the defined auto completion task, finding that
while current off-the-shelf models perform fairly, there is still much room for
improvement, mainly in ranking of the generated suggestions. We provide
insights for practitioners working on this task and open new research
directions for researchers in the field. We release our framework to serve as a
foundation for future research.
 | 2024-12-24T12:03:36Z | [Link](http://arxiv.org/abs/2412.18377v1) |
| Other | Defining and Detecting the Defects of the Large Language Model-based
  Autonomous Agents |   AI agents are systems capable of perceiving their environment, autonomously
planning and executing tasks. Recent advancements in LLM have introduced a
transformative paradigm for AI agents, enabling them to interact with external
resources and tools through prompts. In such agents, the workflow integrates
developer-written code, which manages framework construction and logic control,
with LLM-generated natural language that enhances dynamic decision-making and
interaction. However, discrepancies between developer-implemented logic and the
dynamically generated content of LLMs in terms of behavior and expected
outcomes can lead to defects, such as tool invocation failures and task
execution errors. These issues introduce specific risks, leading to various
defects in LLM-based AI Agents, such as service interruptions. Despite the
importance of these issues, there is a lack of systematic work that focuses on
analyzing LLM-based AI Agents to uncover defects in their code. In this paper,
we present the first study focused on identifying and detecting defects in LLM
Agents. We collected and analyzed 6,854 relevant posts from StackOverflow to
define 8 types of agent defects. For each type, we provided detailed
descriptions with an example. Then, we designed a static analysis tool, named
Agentable, to detect the defects. Agentable leverages Code Property Graphs and
LLMs to analyze Agent workflows by efficiently identifying specific code
patterns and analyzing natural language descriptions. To evaluate Agentable, we
constructed two datasets: AgentSet, consists of 84 real-world Agents, and
AgentTest, which contains 78 Agents specifically designed to include various
types of defects. Our results show that Agentable achieved an overall accuracy
of 88.79% and a recall rate of 91.03%. Furthermore, our analysis reveals the
889 defects of the AgentSet, highlighting the prevalence of these defects.
 | 2024-12-24T11:54:14Z | [Link](http://arxiv.org/abs/2412.18371v1) |
| Other | Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology
  Dataset |   The field of machine translation has achieved significant advancements, yet
domain-specific terminology translation, particularly in AI, remains
challenging. We introduced GIST, a large-scale multilingual AI terminology
dataset containing 5K terms extracted from top AI conference papers spanning
2000 to 2023. The terms were translated into Arabic, Chinese, French, Japanese,
and Russian using a hybrid framework that combines LLMs for extraction with
human expertise for translation. The dataset's quality was benchmarked against
existing resources, demonstrating superior translation accuracy through
crowdsourced evaluation. GIST was integrated into translation workflows using
post-translation refinement methods that required no retraining, where LLM
prompting consistently improved BLEU and COMET scores. A web demonstration on
the ACL Anthology platform highlights its practical application, showcasing
improved accessibility for non-English speakers. This work aims to address
critical gaps in AI terminology resources and fosters global inclusivity and
collaboration in AI research.
 | 2024-12-24T11:50:18Z | [Link](http://arxiv.org/abs/2412.18367v1) |
| Other | Multi-Agents Based on Large Language Models for Knowledge-based Visual
  Question Answering |   Large Language Models (LLMs) have achieved impressive results in
knowledge-based Visual Question Answering (VQA). However existing methods still
have challenges: the inability to use external tools autonomously, and the
inability to work in teams. Humans tend to know whether they need to use
external tools when they encounter a new question, e.g., they tend to be able
to give a direct answer to a familiar question, whereas they tend to use tools
such as search engines when they encounter an unfamiliar question. In addition,
humans also tend to collaborate and discuss with others to get better answers.
Inspired by this, we propose the multi-agent voting framework. We design three
LLM-based agents that simulate different levels of staff in a team, and assign
the available tools according to the levels. Each agent provides the
corresponding answer, and finally all the answers provided by the agents are
voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our
approach outperforms other baselines by 2.2 and 1.0, respectively.
 | 2024-12-24T11:24:56Z | [Link](http://arxiv.org/abs/2412.18351v1) |
| Other | M-Ped: Multi-Prompt Ensemble Decoding for Large Language Models |   With the widespread application of Large Language Models (LLMs) in the field
of Natural Language Processing (NLP), enhancing their performance has become a
research hotspot. This paper presents a novel multi-prompt ensemble decoding
approach designed to bolster the generation quality of LLMs by leveraging the
aggregation of outcomes from multiple prompts. Given a unique input $X$, we
submit $n$ variations of prompts with $X$ to LLMs in batch mode to decode and
derive probability distributions. For each token prediction, we calculate the
ensemble probability by averaging the $n$ probability distributions within the
batch, utilizing this aggregated probability to generate the token. This
technique is dubbed Inner-Batch Ensemble. To facilitate efficient batch
inference, we implement a Left-Padding strategy to maintain uniform input
lengths across the n prompts. Through extensive experimentation on diverse NLP
tasks, including machine translation, code generation, and text simplification,
we demonstrate the efficacy of our method in enhancing LLM performance. The
results show substantial improvements in BLEU scores, pass@$k$ rates, and LENS
metrics over conventional methods.
 | 2024-12-24T09:06:58Z | [Link](http://arxiv.org/abs/2412.18299v1) |
| Other | Quo Vadis, Anomaly Detection? LLMs and VLMs in the Spotlight |   Video anomaly detection (VAD) has witnessed significant advancements through
the integration of large language models (LLMs) and vision-language models
(VLMs), addressing critical challenges such as interpretability, temporal
reasoning, and generalization in dynamic, open-world scenarios. This paper
presents an in-depth review of cutting-edge LLM-/VLM-based methods in 2024,
focusing on four key aspects: (i) enhancing interpretability through semantic
insights and textual explanations, making visual anomalies more understandable;
(ii) capturing intricate temporal relationships to detect and localize dynamic
anomalies across video frames; (iii) enabling few-shot and zero-shot detection
to minimize reliance on large, annotated datasets; and (iv) addressing
open-world and class-agnostic anomalies by using semantic understanding and
motion features for spatiotemporal coherence. We highlight their potential to
redefine the landscape of VAD. Additionally, we explore the synergy between
visual and textual modalities offered by LLMs and VLMs, highlighting their
combined strengths and proposing future directions to fully exploit the
potential in enhancing video anomaly detection.
 | 2024-12-24T09:05:37Z | [Link](http://arxiv.org/abs/2412.18298v1) |
| Other | Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases |   The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in
several real-world services triggers severe concerns about their security. A
RAG system improves the generative capabilities of a Large Language Models
(LLM) by a retrieval mechanism which operates on a private knowledge base,
whose unintended exposure could lead to severe consequences, including breaches
of private and sensitive information. This paper presents a black-box attack to
force a RAG system to leak its private knowledge base which, differently from
existing approaches, is adaptive and automatic. A relevance-based mechanism and
an attacker-side open-source LLM favor the generation of effective queries to
leak most of the (hidden) knowledge base. Extensive experimentation proves the
quality of the proposed algorithm in different RAG pipelines and domains,
comparing to very recent related approaches, which turn out to be either not
fully black-box, not adaptive, or not based on open-source models. The findings
from our study remark the urgent need for more robust privacy safeguards in the
design and deployment of RAG systems.
 | 2024-12-24T09:03:57Z | [Link](http://arxiv.org/abs/2412.18295v1) |
| Other | DeepCRCEval: Revisiting the Evaluation of Code Review Comment Generation |   Code review is a vital but demanding aspect of software development,
generating significant interest in automating review comments. Traditional
evaluation methods for these comments, primarily based on text similarity, face
two major challenges: inconsistent reliability of human-authored comments in
open-source projects and the weak correlation of text similarity with
objectives like enhancing code quality and detecting defects.
  This study empirically analyzes benchmark comments using a novel set of
criteria informed by prior research and developer interviews. We then similarly
revisit the evaluation of existing methodologies. Our evaluation framework,
DeepCRCEval, integrates human evaluators and Large Language Models (LLMs) for a
comprehensive reassessment of current techniques based on the criteria set.
Besides, we also introduce an innovative and efficient baseline, LLM-Reviewer,
leveraging the few-shot learning capabilities of LLMs for a target-oriented
comparison.
  Our research highlights the limitations of text similarity metrics, finding
that less than 10% of benchmark comments are high quality for automation. In
contrast, DeepCRCEval effectively distinguishes between high and low-quality
comments, proving to be a more reliable evaluation mechanism. Incorporating LLM
evaluators into DeepCRCEval significantly boosts efficiency, reducing time and
cost by 88.78% and 90.32%, respectively. Furthermore, LLM-Reviewer demonstrates
significant potential of focusing task real targets in comment generation.
 | 2024-12-24T08:53:54Z | [Link](http://arxiv.org/abs/2412.18291v1) |
| Other | Improving Multi-Step Reasoning Abilities of Large Language Models with
  Direct Advantage Policy Optimization |   The role of reinforcement learning (RL) in enhancing the reasoning of large
language models (LLMs) is becoming increasingly significant. Despite the
success of RL in many scenarios, there are still many challenges in improving
the reasoning of LLMs. One challenge is the sparse reward, which makes
optimization difficult for RL and necessitates a large amount of data samples.
Another challenge stems from the inherent instability of RL, particularly when
using Actor-Critic (AC) methods to derive optimal policies, which often leads
to unstable training processes. To address these issues, we introduce Direct
Advantage Policy Optimization (DAPO), an novel step-level offline RL algorithm.
Unlike standard alignment that rely solely outcome rewards to optimize policies
(such as DPO), DAPO employs a critic function to predict the reasoning accuracy
at each step, thereby generating dense signals to refine the generation
strategy. Additionally, the Actor and Critic components in DAPO are trained
independently, avoiding the co-training instability observed in standard AC
algorithms like PPO. We train DAPO on mathematical and code query datasets and
then evaluate its performance on multiple benchmarks. Our results show that
DAPO can effectively enhance the mathematical and code capabilities on both SFT
models and RL models, demonstrating the effectiveness of DAPO.
 | 2024-12-24T08:39:35Z | [Link](http://arxiv.org/abs/2412.18279v1) |
| Other | GenAI Content Detection Task 2: AI vs. Human -- Academic Essay
  Authenticity Challenge |   This paper presents a comprehensive overview of the first edition of the
Academic Essay Authenticity Challenge, organized as part of the GenAI Content
Detection shared tasks collocated with COLING 2025. This challenge focuses on
detecting machine-generated vs. human-authored essays for academic purposes.
The task is defined as follows: "Given an essay, identify whether it is
generated by a machine or authored by a human.'' The challenge involves two
languages: English and Arabic. During the evaluation phase, 25 teams submitted
systems for English and 21 teams for Arabic, reflecting substantial interest in
the task. Finally, seven teams submitted system description papers. The
majority of submissions utilized fine-tuned transformer-based models, with one
team employing Large Language Models (LLMs) such as Llama 2 and Llama 3. This
paper outlines the task formulation, details the dataset construction process,
and explains the evaluation framework. Additionally, we present a summary of
the approaches adopted by participating teams. Nearly all submitted systems
outperformed the n-gram-based baseline, with the top-performing systems
achieving F1 scores exceeding 0.98 for both languages, indicating significant
progress in the detection of machine-generated text.
 | 2024-12-24T08:33:44Z | [Link](http://arxiv.org/abs/2412.18274v1) |
| Other | Annotating References to Mythological Entities in French Literature |   In this paper, we explore the relevance of large language models (LLMs) for
annotating references to Roman and Greek mythological entities in modern and
contemporary French literature. We present an annotation scheme and demonstrate
that recent LLMs can be directly applied to follow this scheme effectively,
although not without occasionally making significant analytical errors.
Additionally, we show that LLMs (and, more specifically, ChatGPT) are capable
of offering interpretative insights into the use of mythological references by
literary authors. However, we also find that LLMs struggle to accurately
identify relevant passages in novels (when used as an information retrieval
engine), often hallucinating and generating fabricated examples-an issue that
raises significant ethical concerns. Nonetheless, when used carefully, LLMs
remain valuable tools for performing annotations with high accuracy, especially
for tasks that would be difficult to annotate comprehensively on a large scale
through manual methods alone.
 | 2024-12-24T08:29:00Z | [Link](http://arxiv.org/abs/2412.18270v1) |
| Other | Investigating Large Language Models for Code Vulnerability Detection: An
  Experimental Study |   Code vulnerability detection (CVD) is essential for addressing and preventing
system security issues, playing a crucial role in ensuring software security.
Previous learning-based vulnerability detection methods rely on either
fine-tuning medium-size sequence models or training smaller neural networks
from scratch. Recent advancements in large pre-trained language models (LLMs)
have showcased remarkable capabilities in various code intelligence tasks
including code understanding and generation. However, the effectiveness of LLMs
in detecting code vulnerabilities is largely under-explored. This work aims to
investigate the gap by fine-tuning LLMs for the CVD task, involving four
widely-used open-source LLMs. We also implement other five previous graph-based
or medium-size sequence models for comparison. Experiments are conducted on
five commonly-used CVD datasets, including both the part of short samples and
long samples. In addition, we conduct quantitative experiments to investigate
the class imbalance issue and the model's performance on samples of different
lengths, which are rarely studied in previous works. To better facilitate
communities, we open-source all codes and resources of this study in
https://github.com/SakiRinn/LLM4CVD and
https://huggingface.co/datasets/xuefen/VulResource.
 | 2024-12-24T08:20:29Z | [Link](http://arxiv.org/abs/2412.18260v1) |
| Other | An Automatic Graph Construction Framework based on Large Language Models
  for Recommendation |   Graph neural networks (GNNs) have emerged as state-of-the-art methods to
learn from graph-structured data for recommendation. However, most existing
GNN-based recommendation methods focus on the optimization of model structures
and learning strategies based on pre-defined graphs, neglecting the importance
of the graph construction stage. Earlier works for graph construction usually
rely on speciffic rules or crowdsourcing, which are either too simplistic or
too labor-intensive. Recent works start to utilize large language models (LLMs)
to automate the graph construction, in view of their abundant open-world
knowledge and remarkable reasoning capabilities. Nevertheless, they generally
suffer from two limitations: (1) invisibility of global view (e.g., overlooking
contextual information) and (2) construction inefficiency. To this end, we
introduce AutoGraph, an automatic graph construction framework based on LLMs
for recommendation. Specifically, we first use LLMs to infer the user
preference and item knowledge, which is encoded as semantic vectors. Next, we
employ vector quantization to extract the latent factors from the semantic
vectors. The latent factors are then incorporated as extra nodes to link the
user/item nodes, resulting in a graph with in-depth global-view semantics. We
further design metapath-based message aggregation to effectively aggregate the
semantic and collaborative information. The framework is model-agnostic and
compatible with different backbone models. Extensive experiments on three
real-world datasets demonstrate the efficacy and efffciency of AutoGraph
compared to existing baseline methods. We have deployed AutoGraph in Huawei
advertising platform, and gain a 2.69% improvement on RPM and a 7.31%
improvement on eCPM in the online A/B test. Currently AutoGraph has been used
as the main trafffc model, serving hundreds of millions of people.
 | 2024-12-24T07:51:29Z | [Link](http://arxiv.org/abs/2412.18241v1) |
| Other | Adapting Large Language Models for Improving TCP Fairness over WiFi |   The new transmission control protocol (TCP) relies on Deep Learning (DL) for
prediction and optimization, but requires significant manual effort to design
deep neural networks (DNNs) and struggles with generalization in dynamic
environments. Inspired by the success of large language models (LLMs), this
study proposes TCP-LLM, a novel framework leveraging LLMs for TCP applications.
TCP-LLM utilizes pre-trained knowledge to reduce engineering effort, enhance
generalization, and deliver superior performance across diverse TCP tasks.
Applied to reducing flow unfairness, adapting congestion control, and
preventing starvation, TCP-LLM demonstrates significant improvements over TCP
with minimal fine-tuning.
 | 2024-12-24T06:11:10Z | [Link](http://arxiv.org/abs/2412.18200v1) |
| Other | Robustness-aware Automatic Prompt Optimization |   The performance of Large Language Models (LLMs) is based on the quality of
the prompts and the semantic and structural integrity information of the input
data. However, current prompt generation methods primarily focus on generating
prompts for clean input data, often overlooking the impact of perturbed inputs
on prompt performance. To address this limitation, we propose BATprompt (By
Adversarial Training prompt), a novel method for prompt generation designed to
withstand input perturbations (such as typos in the input). Inspired by
adversarial training techniques, BATprompt demonstrates strong performance on a
variety of perturbed tasks through a two-step process: adversarial perturbation
and iterative optimization on unperturbed input via LLM. Unlike conventional
adversarial attack methods, BATprompt avoids reliance on real gradients or
model parameters. Instead, it leverages the advanced reasoning, language
understanding and self reflection capabilities of LLMs to simulate gradients,
guiding the generation of adversarial perturbations and optimizing prompt
performance. In our experiments, we evaluate BATprompt on multiple datasets
across both language understanding and generation tasks. The results indicate
that BATprompt outperforms existing prompt generation methods, delivering
superior robustness and performance under diverse perturbation scenarios.
 | 2024-12-24T06:05:08Z | [Link](http://arxiv.org/abs/2412.18196v1) |
| Other | VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics
  Manipulation with Long-Horizon Reasoning Tasks |   General-purposed embodied agents are designed to understand the users'
natural instructions or intentions and act precisely to complete universal
tasks. Recently, methods based on foundation models especially
Vision-Language-Action models (VLAs) have shown a substantial potential to
solve language-conditioned manipulation (LCM) tasks well. However, existing
benchmarks do not adequately meet the needs of VLAs and relative algorithms. To
better define such general-purpose tasks in the context of LLMs and advance the
research in VLAs, we present VLABench, an open-source benchmark for evaluating
universal LCM task learning. VLABench provides 100 carefully designed
categories of tasks, with strong randomization in each category of task and a
total of 2000+ objects. VLABench stands out from previous benchmarks in four
key aspects: 1) tasks requiring world knowledge and common sense transfer, 2)
natural language instructions with implicit human intentions rather than
templates, 3) long-horizon tasks demanding multi-step reasoning, and 4)
evaluation of both action policies and language model capabilities. The
benchmark assesses multiple competencies including understanding of
mesh\&texture, spatial relationship, semantic instruction, physical laws,
knowledge transfer and reasoning, etc. To support the downstream finetuning, we
provide high-quality training data collected via an automated framework
incorporating heuristic skills and prior information. The experimental results
indicate that both the current state-of-the-art pretrained VLAs and the
workflow based on VLMs face challenges in our tasks.
 | 2024-12-24T06:03:42Z | [Link](http://arxiv.org/abs/2412.18194v1) |
| Other | TextMatch: Enhancing Image-Text Consistency Through Multimodal
  Optimization |   Text-to-image generative models excel in creating images from text but
struggle with ensuring alignment and consistency between outputs and prompts.
This paper introduces TextMatch, a novel framework that leverages multimodal
optimization to address image-text discrepancies in text-to-image (T2I)
generation and editing. TextMatch employs a scoring strategy powered by large
language models (LLMs) and visual question-answering (VQA) models to evaluate
semantic consistency between prompts and generated images. By integrating
multimodal in-context learning and chain of thought reasoning, our method
dynamically refines prompts through iterative optimization. This process
ensures that the generated images better capture user intent of, resulting in
higher fidelity and relevance. Extensive experiments demonstrate that TextMatch
significantly improves text-image consistency across multiple benchmarks,
establishing a reliable framework for advancing the capabilities of
text-to-image generative models. Our code is available at
https://anonymous.4open.science/r/TextMatch-F55C/.
 | 2024-12-24T05:38:45Z | [Link](http://arxiv.org/abs/2412.18185v1) |
| Other | Molar: Multimodal LLMs with Collaborative Filtering Alignment for
  Enhanced Sequential Recommendation |   Sequential recommendation (SR) systems have evolved significantly over the
past decade, transitioning from traditional collaborative filtering to deep
learning approaches and, more recently, to large language models (LLMs). While
the adoption of LLMs has driven substantial advancements, these models
inherently lack collaborative filtering information, relying primarily on
textual content data neglecting other modalities and thus failing to achieve
optimal recommendation performance. To address this limitation, we propose
Molar, a Multimodal large language sequential recommendation framework that
integrates multiple content modalities with ID information to capture
collaborative signals effectively. Molar employs an MLLM to generate unified
item representations from both textual and non-textual data, facilitating
comprehensive multimodal modeling and enriching item embeddings. Additionally,
it incorporates collaborative filtering signals through a post-alignment
mechanism, which aligns user representations from content-based and ID-based
models, ensuring precise personalization and robust performance. By seamlessly
combining multimodal content with collaborative filtering insights, Molar
captures both user interests and contextual semantics, leading to superior
recommendation accuracy. Extensive experiments validate that Molar
significantly outperforms traditional and LLM-based baselines, highlighting its
strength in utilizing multimodal data and collaborative signals for sequential
recommendation tasks. The source code is available at
https://anonymous.4open.science/r/Molar-8B06/.
 | 2024-12-24T05:23:13Z | [Link](http://arxiv.org/abs/2412.18176v1) |
| Other | INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with
  LLM-based Agent |   Recent advancements have underscored the potential of large language model
(LLM)-based agents in financial decision-making. Despite this progress, the
field currently encounters two main challenges: (1) the lack of a comprehensive
LLM agent framework adaptable to a variety of financial tasks, and (2) the
absence of standardized benchmarks and consistent datasets for assessing agent
performance. To tackle these issues, we introduce \textsc{InvestorBench}, the
first benchmark specifically designed for evaluating LLM-based agents in
diverse financial decision-making contexts. InvestorBench enhances the
versatility of LLM-enabled agents by providing a comprehensive suite of tasks
applicable to different financial products, including single equities like
stocks, cryptocurrencies and exchange-traded funds (ETFs). Additionally, we
assess the reasoning and decision-making capabilities of our agent framework
using thirteen different LLMs as backbone models, across various market
environments and tasks. Furthermore, we have curated a diverse collection of
open-source, multi-modal datasets and developed a comprehensive suite of
environments for financial decision-making. This establishes a highly
accessible platform for evaluating financial agents' performance across various
scenarios.
 | 2024-12-24T05:22:33Z | [Link](http://arxiv.org/abs/2412.18174v1) |
| Other | Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large
  Language Models |   Large Language Models (LLMs) are increasingly being integrated into services
such as ChatGPT to provide responses to user queries. To mitigate potential
harm and prevent misuse, there have been concerted efforts to align the LLMs
with human values and legal compliance by incorporating various techniques,
such as Reinforcement Learning from Human Feedback (RLHF), into the training of
the LLMs. However, recent research has exposed that even aligned LLMs are
susceptible to adversarial manipulations known as Jailbreak Attacks. To address
this challenge, this paper proposes a method called Token Highlighter to
inspect and mitigate the potential jailbreak threats in the user query. Token
Highlighter introduced a concept called Affirmation Loss to measure the LLM's
willingness to answer the user query. It then uses the gradient of Affirmation
Loss for each token in the user query to locate the jailbreak-critical tokens.
Further, Token Highlighter exploits our proposed Soft Removal technique to
mitigate the jailbreak effects of critical tokens via shrinking their token
embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5)
demonstrate that the proposed method can effectively defend against a variety
of Jailbreak Attacks while maintaining competent performance on benign
questions of the AlpacaEval benchmark. In addition, Token Highlighter is a
cost-effective and interpretable defense because it only needs to query the
protected LLM once to compute the Affirmation Loss and can highlight the
critical tokens upon refusal.
 | 2024-12-24T05:10:02Z | [Link](http://arxiv.org/abs/2412.18171v1) |
| Other | KunServe: Elastic and Efficient Large Language Model Serving with
  Parameter-centric Memory Management |   The stateful nature of large language model (LLM) servingcan easily throttle
precious GPU memory under load burstor long-generation requests like
chain-of-thought reasoning,causing latency spikes due to queuing incoming
requests. However, state-of-the-art KVCache centric approaches handleload
spikes by dropping, migrating, or swapping KVCache,which faces an essential
tradeoff between the performance ofongoing vs. incoming requests and thus still
severely violatesSLO.This paper makes a key observation such that model
param-eters are independent of the requests and are replicated acrossGPUs, and
thus proposes a parameter-centric approach byselectively dropping replicated
parameters to leave preciousmemory for requests. However, LLM requires KVCache
tobe saved in bound with model parameters and thus droppingparameters can cause
either huge computation waste or longnetwork delay, affecting all ongoing
requests. Based on the ob-servation that attention operators can be decoupled
from otheroperators, this paper further proposes a novel remote
attentionmechanism through pipeline parallelism so as to serve up-coming
requests with the additional memory borrowed fromparameters on remote GPUs.
This paper further addresses sev-eral other challenges including lively
exchanging KVCachewith incomplete parameters, generating an appropriate
planthat balances memory requirements with cooperative exe-cution overhead, and
seamlessly restoring parameters whenthe throttling has gone. Evaluations show
thatKUNSERVEreduces the tail TTFT of requests under throttling by up to 27.3x
compared to the state-of-the-art.
 | 2024-12-24T05:07:46Z | [Link](http://arxiv.org/abs/2412.18169v1) |
| Other | VISION: A Modular AI Assistant for Natural Human-Instrument Interaction
  at Scientific User Facilities |   Scientific user facilities, such as synchrotron beamlines, are equipped with
a wide array of hardware and software tools that require a codebase for
human-computer-interaction. This often necessitates developers to be involved
to establish connection between users/researchers and the complex
instrumentation. The advent of generative AI presents an opportunity to bridge
this knowledge gap, enabling seamless communication and efficient experimental
workflows. Here we present a modular architecture for the Virtual Scientific
Companion (VISION) by assembling multiple AI-enabled cognitive blocks that each
scaffolds large language models (LLMs) for a specialized task. With VISION, we
performed LLM-based operation on the beamline workstation with low latency and
demonstrated the first voice-controlled experiment at an X-ray scattering
beamline. The modular and scalable architecture allows for easy adaptation to
new instrument and capabilities. Development on natural language-based
scientific experimentation is a building block for an impending future where a
science exocortex -- a synthetic extension to the cognition of scientists --
may radically transform scientific practice and discovery.
 | 2024-12-24T04:37:07Z | [Link](http://arxiv.org/abs/2412.18161v1) |
| Other | scReader: Prompting Large Language Models to Interpret scRNA-seq Data |   Large language models (LLMs) have demonstrated remarkable advancements,
primarily due to their capabilities in modeling the hidden relationships within
text sequences. This innovation presents a unique opportunity in the field of
life sciences, where vast collections of single-cell omics data from multiple
species provide a foundation for training foundational models. However, the
challenge lies in the disparity of data scales across different species,
hindering the development of a comprehensive model for interpreting genetic
data across diverse organisms. In this study, we propose an innovative hybrid
approach that integrates the general knowledge capabilities of LLMs with
domain-specific representation models for single-cell omics data
interpretation. We begin by focusing on genes as the fundamental unit of
representation. Gene representations are initialized using functional
descriptions, leveraging the strengths of mature language models such as
LLaMA-2. By inputting single-cell gene-level expression data with prompts, we
effectively model cellular representations based on the differential expression
levels of genes across various species and cell types. In the experiments, we
constructed developmental cells from humans and mice, specifically targeting
cells that are challenging to annotate. We evaluated our methodology through
basic tasks such as cell annotation and visualization analysis. The results
demonstrate the efficacy of our approach compared to other methods using LLMs,
highlighting significant improvements in accuracy and interoperability. Our
hybrid approach enhances the representation of single-cell data and offers a
robust framework for future research in cross-species genetic analysis.
 | 2024-12-24T04:28:42Z | [Link](http://arxiv.org/abs/2412.18156v1) |
| Other | GeneSUM: Large Language Model-based Gene Summary Extraction |   Emerging topics in biomedical research are continuously expanding, providing
a wealth of information about genes and their function. This rapid
proliferation of knowledge presents unprecedented opportunities for scientific
discovery and formidable challenges for researchers striving to keep abreast of
the latest advancements. One significant challenge is navigating the vast
corpus of literature to extract vital gene-related information, a
time-consuming and cumbersome task. To enhance the efficiency of this process,
it is crucial to address several key challenges: (1) the overwhelming volume of
literature, (2) the complexity of gene functions, and (3) the automated
integration and generation. In response, we propose GeneSUM, a two-stage
automated gene summary extractor utilizing a large language model (LLM). Our
approach retrieves and eliminates redundancy of target gene literature and then
fine-tunes the LLM to refine and streamline the summarization process. We
conducted extensive experiments to validate the efficacy of our proposed
framework. The results demonstrate that LLM significantly enhances the
integration of gene-specific information, allowing more efficient
decision-making in ongoing research.
 | 2024-12-24T04:20:43Z | [Link](http://arxiv.org/abs/2412.18154v1) |
| Other | Are We in the AI-Generated Text World Already? Quantifying and
  Monitoring AIGT on Social Media |   Social media platforms are experiencing a growing presence of AI-Generated
Texts (AIGTs). However, the misuse of AIGTs could have profound implications
for public opinion, such as spreading misinformation and manipulating
narratives. Despite its importance, a systematic study to assess the prevalence
of AIGTs on social media is still lacking. To address this gap, this paper aims
to quantify, monitor, and analyze the AIGTs on online social media platforms.
We first collect a dataset (SM-D) with around 2.4M posts from 3 major social
media platforms: Medium, Quora, and Reddit. Then, we construct a diverse
dataset (AIGTBench) to train and evaluate AIGT detectors. AIGTBench combines
popular open-source datasets and our AIGT datasets generated from social media
texts by 12 LLMs, serving as a benchmark for evaluating mainstream detectors.
With this setup, we identify the best-performing detector (OSM-Det). We then
apply OSM-Det to SM-D to track AIGTs over time and observe different trends of
AI Attribution Rate (AAR) across social media platforms from January 2022 to
October 2024. Specifically, Medium and Quora exhibit marked increases in AAR,
rising from 1.77% to 37.03% and 2.06% to 38.95%, respectively. In contrast,
Reddit shows slower growth, with AAR increasing from 1.31% to 2.45% over the
same period. Our further analysis indicates that AIGTs differ from
human-written texts across several dimensions, including linguistic patterns,
topic distributions, engagement levels, and the follower distribution of
authors. We envision our analysis and findings on AIGTs in social media can
shed light on future research in this domain.
 | 2024-12-24T04:04:54Z | [Link](http://arxiv.org/abs/2412.18148v1) |
| Other | LSAQ: Layer-Specific Adaptive Quantization for Large Language Model
  Deployment |   As large language models (LLMs) demonstrate exceptional performance across
various domains, the deployment of these models on edge devices has emerged as
a new trend. Quantization techniques, which reduce the size and memory
footprint of LLMs, are effective for enabling deployment on
resource-constrained edge devices. However, existing one-size-fits-all
quantization methods often fail to dynamically adjust the memory consumption of
LLMs based on specific hardware characteristics and usage scenarios. To address
this limitation, we propose LSAQ (Layer-Specific Adaptive Quantization), a
system for adaptive quantization and dynamic deployment of LLMs based on layer
importance. LSAQ evaluates layer importance by constructing top-k token sets
from the inputs and outputs of each layer and calculating their Jaccard
coefficient. Using this evaluation, the system adaptively adjusts quantization
strategies in real time according to the resource availability of edge devices,
assigning different precision levels to layers of varying importance. This
approach significantly reduces the storage requirements of LLMs while
maintaining model performance, enabling efficient deployment across diverse
hardware platforms and usage scenarios.
 | 2024-12-24T03:43:15Z | [Link](http://arxiv.org/abs/2412.18135v1) |
| Other | AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation |   Large language models (LLMs) have brought exciting new advances to mobile UI
agents, a long-standing research field that aims to complete arbitrary natural
language tasks through mobile UI interactions. However, existing UI agents
usually demand high reasoning capabilities of powerful large models that are
difficult to be deployed locally on end-users' devices, which raises huge
concerns about user privacy and centralized serving cost. One way to reduce the
required model size is to customize a smaller domain-specific model with
high-quality training data, e.g. large-scale human demonstrations of diverse
types of apps and tasks, while such datasets are extremely difficult to obtain.
Inspired by the remarkable coding abilities of recent small language models
(SLMs), we propose to convert the UI task automation problem to a code
generation problem, which can be effectively solved by an on-device SLM and
efficiently executed with an on-device code interpreter. Unlike normal coding
tasks that can be extensively pretrained with public datasets, generating UI
automation code is challenging due to the diversity, complexity, and
variability of target apps. Therefore, we adopt a document-centered approach
that automatically builds fine-grained API documentation for each app and
generates diverse task samples based on this documentation. By guiding the
agent with the synthetic documents and task samples, it learns to generate
precise and efficient scripts to complete unseen tasks. Based on detailed
comparisons with state-of-the-art mobile UI agents, our approach effectively
improves the mobile task automation with significantly higher success rates and
lower latency/token consumption. Code will be open-sourced.
 | 2024-12-24T02:54:56Z | [Link](http://arxiv.org/abs/2412.18116v1) |
| Other | AIGT: AI Generative Table Based on Prompt |   Tabular data, which accounts for over 80% of enterprise data assets, is vital
in various fields. With growing concerns about privacy protection and
data-sharing restrictions, generating high-quality synthetic tabular data has
become essential. Recent advancements show that large language models (LLMs)
can effectively gener-ate realistic tabular data by leveraging semantic
information and overcoming the challenges of high-dimensional data that arise
from one-hot encoding. However, current methods do not fully utilize the rich
information available in tables. To address this, we introduce AI Generative
Table (AIGT) based on prompt enhancement, a novel approach that utilizes meta
data information, such as table descriptions and schemas, as prompts to
generate ultra-high quality synthetic data. To overcome the token limit
constraints of LLMs, we propose long-token partitioning algorithms that enable
AIGT to model tables of any scale. AIGT achieves state-of-the-art performance
on 14 out of 20 public datasets and two real industry datasets within the
Alipay risk control system.
 | 2024-12-24T02:51:06Z | [Link](http://arxiv.org/abs/2412.18111v1) |
| Other | SlimGPT: Layer-wise Structured Pruning for Large Language Models |   Large language models (LLMs) have garnered significant attention for their
remarkable capabilities across various domains, whose vast parameter scales
present challenges for practical deployment. Structured pruning is an effective
method to balance model performance with efficiency, but performance
restoration under computational resource constraints is a principal challenge
in pruning LLMs. Therefore, we present a low-cost and fast structured pruning
method for LLMs named SlimGPT based on the Optimal Brain Surgeon framework. We
propose Batched Greedy Pruning for rapid and near-optimal pruning, which
enhances the accuracy of head-wise pruning error estimation through grouped
Cholesky decomposition and improves the pruning efficiency of FFN via Dynamic
Group Size, thereby achieving approximate local optimal pruning results within
one hour. Besides, we explore the limitations of layer-wise pruning from the
perspective of error accumulation and propose Incremental Pruning Ratio, a
non-uniform pruning strategy to reduce performance degradation. Experimental
results on the LLaMA benchmark show that SlimGPT outperforms other methods and
achieves state-of-the-art results.
 | 2024-12-24T02:49:50Z | [Link](http://arxiv.org/abs/2412.18110v1) |
| Other | Unveiling Visual Perception in Language Models: An Attention Head
  Analysis Approach |   Recent advancements in Multimodal Large Language Models (MLLMs) have
demonstrated remarkable progress in visual understanding. This impressive leap
raises a compelling question: how can language models, initially trained solely
on linguistic data, effectively interpret and process visual content? This
paper aims to address this question with systematic investigation across 4
model families and 4 model scales, uncovering a unique class of attention heads
that focus specifically on visual content. Our analysis reveals a strong
correlation between the behavior of these attention heads, the distribution of
attention weights, and their concentration on visual tokens within the input.
These findings enhance our understanding of how LLMs adapt to multimodal tasks,
demonstrating their potential to bridge the gap between textual and visual
understanding. This work paves the way for the development of AI systems
capable of engaging with diverse modalities.
 | 2024-12-24T02:31:24Z | [Link](http://arxiv.org/abs/2412.18108v1) |
| Other | Tackling the Dynamicity in a Production LLM Serving System with SOTA
  Optimizations via Hybrid Prefill/Decode/Verify Scheduling on Efficient
  Meta-kernels |   Meeting growing demands for low latency and cost efficiency in
production-grade large language model (LLM) serving systems requires
integrating advanced optimization techniques. However, dynamic and
unpredictable input-output lengths of LLM, compounded by these optimizations,
exacerbate the issues of workload variability, making it difficult to maintain
high efficiency on AI accelerators, especially DSAs with tile-based programming
models. To address this challenge, we introduce XY-Serve, a versatile, Ascend
native, end-to-end production LLM-serving system. The core idea is an
abstraction mechanism that smooths out the workload variability by decomposing
computations into unified, hardware-friendly, fine-grained meta primitives. For
attention, we propose a meta-kernel that computes the basic pattern of
matmul-softmax-matmul with architectural-aware tile sizes. For GEMM, we
introduce a virtual padding scheme that adapts to dynamic shape changes while
using highly efficient GEMM primitives with assorted fixed tile sizes. XY-Serve
sits harmoniously with vLLM. Experimental results show up to 89% end-to-end
throughput improvement compared with current publicly available baselines on
Ascend NPUs. Additionally, our approach outperforms existing GEMM (average
14.6% faster) and attention (average 21.5% faster) kernels relative to existing
libraries. While the work is Ascend native, we believe the approach can be
readily applicable to SIMT architectures as well.
 | 2024-12-24T02:27:44Z | [Link](http://arxiv.org/abs/2412.18106v1) |
| Other | EvoPat: A Multi-LLM-based Patents Summarization and Analysis Agent |   The rapid growth of scientific techniques and knowledge is reflected in the
exponential increase in new patents filed annually. While these patents drive
innovation, they also present significant burden for researchers and engineers,
especially newcomers. To avoid the tedious work of navigating a vast and
complex landscape to identify trends and breakthroughs, researchers urgently
need efficient tools to summarize, evaluate, and contextualize patents,
revealing their innovative contributions and underlying scientific
principles.To address this need, we present EvoPat, a multi-LLM-based patent
agent designed to assist users in analyzing patents through Retrieval-Augmented
Generation (RAG) and advanced search strategies. EvoPat leverages multiple
Large Language Models (LLMs), each performing specialized roles such as
planning, identifying innovations, and conducting comparative evaluations. The
system integrates data from local databases, including patents, literature,
product catalogous, and company repositories, and online searches to provide
up-to-date insights. The ability to collect information not included in
original database automatically is also implemented. Through extensive testing
in the natural language processing (NLP) domain, we demonstrate that EvoPat
outperforms GPT-4 in tasks such as patent summarization, comparative analysis,
and technical evaluation. EvoPat represents a significant step toward creating
AI-powered tools that empower researchers and engineers to efficiently navigate
the complexities of the patent landscape.
 | 2024-12-24T02:21:09Z | [Link](http://arxiv.org/abs/2412.18100v1) |
| Other | Generating Traffic Scenarios via In-Context Learning to Learn Better
  Motion Planner |   Motion planning is a crucial component in autonomous driving.
State-of-the-art motion planners are trained on meticulously curated datasets,
which are not only expensive to annotate but also insufficient in capturing
rarely seen critical scenarios. Failing to account for such scenarios poses a
significant risk to motion planners and may lead to incidents during testing.
An intuitive solution is to manually compose such scenarios by programming and
executing a simulator (e.g., CARLA). However, this approach incurs substantial
human costs. Motivated by this, we propose an inexpensive method for generating
diverse critical traffic scenarios to train more robust motion planners. First,
we represent traffic scenarios as scripts, which are then used by the simulator
to generate traffic scenarios. Next, we develop a method that accepts
user-specified text descriptions, which a Large Language Model (LLM) translates
into scripts using in-context learning. The output scripts are sent to the
simulator that produces the corresponding traffic scenarios. As our method can
generate abundant safety-critical traffic scenarios, we use them as synthetic
training data for motion planners. To demonstrate the value of generated
scenarios, we train existing motion planners on our synthetic data, real-world
datasets, and a combination of both. Our experiments show that motion planners
trained with our data significantly outperform those trained solely on
real-world data, showing the usefulness of our synthetic data and the
effectiveness of our data generation method. Our source code is available at
https://ezharjan.github.io/AutoSceneGen.
 | 2024-12-24T01:52:19Z | [Link](http://arxiv.org/abs/2412.18086v1) |
| Other | Property Enhanced Instruction Tuning for Multi-task Molecule Generation
  with Large Language Models |   Large language models (LLMs) are widely applied in various natural language
processing tasks such as question answering and machine translation. However,
due to the lack of labeled data and the difficulty of manual annotation for
biochemical properties, the performance for molecule generation tasks is still
limited, especially for tasks involving multi-properties constraints. In this
work, we present a two-step framework PEIT (Property Enhanced Instruction
Tuning) to improve LLMs for molecular-related tasks. In the first step, we use
textual descriptions, SMILES, and biochemical properties as multimodal inputs
to pre-train a model called PEIT-GEN, by aligning multi-modal representations
to synthesize instruction data. In the second step, we fine-tune existing
open-source LLMs with the synthesized data, the resulting PEIT-LLM can handle
molecule captioning, text-based molecule generation, molecular property
prediction, and our newly proposed multi-constraint molecule generation tasks.
Experimental results show that our pre-trained PEIT-GEN outperforms MolT5 and
BioT5 in molecule captioning, demonstrating modalities align well between
textual descriptions, structures, and biochemical properties. Furthermore,
PEIT-LLM shows promising improvements in multi-task molecule generation,
proving the scalability of the PEIT framework for various molecular tasks. We
release the code, constructed instruction data, and model checkpoints in
https://github.com/chenlong164/PEIT.
 | 2024-12-24T01:48:07Z | [Link](http://arxiv.org/abs/2412.18084v1) |
| Other | MMFactory: A Universal Solution Search Engine for Vision-Language Tasks |   With advances in foundational and vision-language models, and effective
fine-tuning techniques, a large number of both general and special-purpose
models have been developed for a variety of visual tasks. Despite the
flexibility and accessibility of these models, no single model is able to
handle all tasks and/or applications that may be envisioned by potential users.
Recent approaches, such as visual programming and multimodal LLMs with
integrated tools aim to tackle complex visual tasks, by way of program
synthesis. However, such approaches overlook user constraints (e.g.,
performance / computational needs), produce test-time sample-specific solutions
that are difficult to deploy, and, sometimes, require low-level instructions
that maybe beyond the abilities of a naive user. To address these limitations,
we introduce MMFactory, a universal framework that includes model and metrics
routing components, acting like a solution search engine across various
available models. Based on a task description and few sample input-output pairs
and (optionally) resource and/or performance constraints, MMFactory can suggest
a diverse pool of programmatic solutions by instantiating and combining
visio-lingual tools from its model repository. In addition to synthesizing
these solutions, MMFactory also proposes metrics and benchmarks performance /
resource characteristics, allowing users to pick a solution that meets their
unique design constraints. From the technical perspective, we also introduced a
committee-based solution proposer that leverages multi-agent LLM conversation
to generate executable, diverse, universal, and robust solutions for the user.
Experimental results show that MMFactory outperforms existing methods by
delivering state-of-the-art solutions tailored to user problem specifications.
Project page is available at https://davidhalladay.github.io/mmfactory_demo.
 | 2024-12-24T00:59:16Z | [Link](http://arxiv.org/abs/2412.18072v1) |
| Other | LMRPA: Large Language Model-Driven Efficient Robotic Process Automation
  for OCR |   This paper introduces LMRPA, a novel Large Model-Driven Robotic Process
Automation (RPA) model designed to greatly improve the efficiency and speed of
Optical Character Recognition (OCR) tasks. Traditional RPA platforms often
suffer from performance bottlenecks when handling high-volume repetitive
processes like OCR, leading to a less efficient and more time-consuming
process. LMRPA allows the integration of Large Language Models (LLMs) to
improve the accuracy and readability of extracted text, overcoming the
challenges posed by ambiguous characters and complex text structures.Extensive
benchmarks were conducted comparing LMRPA to leading RPA platforms, including
UiPath and Automation Anywhere, using OCR engines like Tesseract and DocTR. The
results are that LMRPA achieves superior performance, cutting the processing
times by up to 52\%. For instance, in Batch 2 of the Tesseract OCR task, LMRPA
completed the process in 9.8 seconds, where UiPath finished in 18.1 seconds and
Automation Anywhere finished in 18.7 seconds. Similar improvements were
observed with DocTR, where LMRPA outperformed other automation tools conducting
the same process by completing tasks in 12.7 seconds, while competitors took
over 20 seconds to do the same. These findings highlight the potential of LMRPA
to revolutionize OCR-driven automation processes, offering a more efficient and
effective alternative solution to the existing state-of-the-art RPA models.
 | 2024-12-24T00:21:36Z | [Link](http://arxiv.org/abs/2412.18063v1) |
| Other | Lla-VAP: LSTM Ensemble of Llama and VAP for Turn-Taking Prediction |   Turn-taking prediction is the task of anticipating when the speaker in a
conversation will yield their turn to another speaker to begin speaking. This
project expands on existing strategies for turn-taking prediction by employing
a multi-modal ensemble approach that integrates large language models (LLMs)
and voice activity projection (VAP) models. By combining the linguistic
capabilities of LLMs with the temporal precision of VAP models, we aim to
improve the accuracy and efficiency of identifying TRPs in both scripted and
unscripted conversational scenarios. Our methods are evaluated on the
In-Conversation Corpus (ICC) and Coached Conversational Preference Elicitation
(CCPE) datasets, highlighting the strengths and limitations of current models
while proposing a potentially more robust framework for enhanced prediction.
 | 2024-12-24T00:20:38Z | [Link](http://arxiv.org/abs/2412.18061v1) |
| Other | An Ensemble Approach to Short-form Video Quality Assessment Using
  Multimodal LLM |   The rise of short-form videos, characterized by diverse content, editing
styles, and artifacts, poses substantial challenges for learning-based blind
video quality assessment (BVQA) models. Multimodal large language models
(MLLMs), renowned for their superior generalization capabilities, present a
promising solution. This paper focuses on effectively leveraging a pretrained
MLLM for short-form video quality assessment, regarding the impacts of
pre-processing and response variability, and insights on combining the MLLM
with BVQA models. We first investigated how frame pre-processing and sampling
techniques influence the MLLM's performance. Then, we introduced a lightweight
learning-based ensemble method that adaptively integrates predictions from the
MLLM and state-of-the-art BVQA models. Our results demonstrated superior
generalization performance with the proposed ensemble approach. Furthermore,
the analysis of content-aware ensemble weights highlighted that some video
characteristics are not fully represented by existing BVQA models, revealing
potential directions to improve BVQA models further.
 | 2024-12-24T00:13:10Z | [Link](http://arxiv.org/abs/2412.18060v1) |
| Other | Factuality or Fiction? Benchmarking Modern LLMs on Ambiguous QA with
  Citations |   Benchmarking modern large language models (LLMs) on complex and realistic
tasks is critical to advancing their development. In this work, we evaluate the
factual accuracy and citation performance of state-of-the-art LLMs on the task
of Question Answering (QA) in ambiguous settings with source citations. Using
three recently published datasets-DisentQA-DupliCite, DisentQA-ParaCite, and
AmbigQA-Cite-featuring a range of real-world ambiguities, we analyze the
performance of two leading LLMs, GPT-4o-mini and Claude-3.5. Our results show
that larger, recent models consistently predict at least one correct answer in
ambiguous contexts but fail to handle cases with multiple valid answers.
Additionally, all models perform equally poorly in citation generation, with
citation accuracy consistently at 0. However, introducing conflict-aware
prompting leads to large improvements, enabling models to better address
multiple valid answers and improve citation accuracy, while maintaining their
ability to predict correct answers. These findings highlight the challenges and
opportunities in developing LLMs that can handle ambiguity and provide reliable
source citations. Our benchmarking study provides critical insights and sets a
foundation for future improvements in trustworthy and interpretable QA systems.
 | 2024-12-23T23:55:19Z | [Link](http://arxiv.org/abs/2412.18051v1) |
| Other | More than Chit-Chat: Developing Robots for Small-Talk Interactions |   Beyond mere formality, small talk plays a pivotal role in social dynamics,
serving as a verbal handshake for building rapport and understanding. For
conversational AI and social robots, the ability to engage in small talk
enhances their perceived sociability, leading to more comfortable and natural
user interactions. In this study, we evaluate the capacity of current Large
Language Models (LLMs) to drive the small talk of a social robot and identify
key areas for improvement. We introduce a novel method that autonomously
generates feedback and ensures LLM-generated responses align with small talk
conventions. Through several evaluations -- involving chatbot interactions and
human-robot interactions -- we demonstrate the system's effectiveness in
guiding LLM-generated responses toward realistic, human-like, and natural
small-talk exchanges.
 | 2024-12-23T22:35:38Z | [Link](http://arxiv.org/abs/2412.18023v1) |
| Other | Trustworthy and Efficient LLMs Meet Databases |   In the rapidly evolving AI era with large language models (LLMs) at the core,
making LLMs more trustworthy and efficient, especially in output generation
(inference), has gained significant attention. This is to reduce plausible but
faulty LLM outputs (a.k.a hallucinations) and meet the highly increased
inference demands. This tutorial explores such efforts and makes them
transparent to the database community. Understanding these efforts is essential
in harnessing LLMs in database tasks and adapting database techniques to LLMs.
Furthermore, we delve into the synergy between LLMs and databases, highlighting
new opportunities and challenges in their intersection. This tutorial aims to
share with database researchers and practitioners essential concepts and
strategies around LLMs, reduce the unfamiliarity of LLMs, and inspire joining
in the intersection between LLMs and databases.
 | 2024-12-23T22:34:40Z | [Link](http://arxiv.org/abs/2412.18022v1) |
| Other | StructTest: Benchmarking LLMs' Reasoning through Compositional
  Structured Outputs |   The rapid development of large language models (LLMs) necessitates robust,
unbiased, and scalable methods for evaluating their capabilities. However,
human annotations are expensive to scale, model-based evaluations are prone to
biases in answer style, while target-answer-based benchmarks are vulnerable to
data contamination and cheating. To address these limitations, we propose
StructTest, a novel benchmark that evaluates LLMs on their ability to produce
compositionally specified structured outputs as an unbiased, cheap-to-run and
difficult-to-cheat measure. The evaluation is done deterministically by a
rule-based evaluator, which can be easily extended to new tasks. By testing
structured outputs across diverse task domains -- including Summarization,
Code, HTML and Math -- we demonstrate that StructTest serves as a good proxy
for general reasoning abilities, as producing structured outputs often requires
internal logical reasoning. We believe that StructTest offers a critical,
complementary approach to objective and robust model evaluation.
 | 2024-12-23T22:08:40Z | [Link](http://arxiv.org/abs/2412.18011v1) |
| Other | LMV-RPA: Large Model Voting-based Robotic Process Automation |   Automating high-volume unstructured data processing is essential for
operational efficiency. Optical Character Recognition (OCR) is critical but
often struggles with accuracy and efficiency in complex layouts and ambiguous
text. These challenges are especially pronounced in large-scale tasks requiring
both speed and precision. This paper introduces LMV-RPA, a Large Model
Voting-based Robotic Process Automation system to enhance OCR workflows.
LMV-RPA integrates outputs from OCR engines such as Paddle OCR, Tesseract OCR,
Easy OCR, and DocTR with Large Language Models (LLMs) like LLaMA 3 and
Gemini-1.5-pro. Using a majority voting mechanism, it processes OCR outputs
into structured JSON formats, improving accuracy, particularly in complex
layouts. The multi-phase pipeline processes text extracted by OCR engines
through LLMs, combining results to ensure the most accurate outputs. LMV-RPA
achieves 99 percent accuracy in OCR tasks, surpassing baseline models with 94
percent, while reducing processing time by 80 percent. Benchmark evaluations
confirm its scalability and demonstrate that LMV-RPA offers a faster, more
reliable, and efficient solution for automating large-scale document processing
tasks.
 | 2024-12-23T20:28:22Z | [Link](http://arxiv.org/abs/2412.17965v1) |
| Other | Dynamic Multi-Agent Orchestration and Retrieval for Multi-Source
  Question-Answer Systems using Large Language Models |   We propose a methodology that combines several advanced techniques in Large
Language Model (LLM) retrieval to support the development of robust,
multi-source question-answer systems. This methodology is designed to integrate
information from diverse data sources, including unstructured documents (PDFs)
and structured databases, through a coordinated multi-agent orchestration and
dynamic retrieval approach. Our methodology leverages specialized agents-such
as SQL agents, Retrieval-Augmented Generation (RAG) agents, and router agents -
that dynamically select the most appropriate retrieval strategy based on the
nature of each query. To further improve accuracy and contextual relevance, we
employ dynamic prompt engineering, which adapts in real time to query-specific
contexts. The methodology's effectiveness is demonstrated within the domain of
Contract Management, where complex queries often require seamless interaction
between unstructured and structured data. Our results indicate that this
approach enhances response accuracy and relevance, offering a versatile and
scalable framework for developing question-answer systems that can operate
across various domains and data sources.
 | 2024-12-23T20:28:20Z | [Link](http://arxiv.org/abs/2412.17964v1) |
| Other | Path-of-Thoughts: Extracting and Following Paths for Robust Relational
  Reasoning with Large Language Models |   Large language models (LLMs) possess vast semantic knowledge but often
struggle with complex reasoning tasks, particularly in relational reasoning
problems such as kinship or spatial reasoning. In this paper, we present
Path-of-Thoughts (PoT), a novel framework designed to tackle relation reasoning
by decomposing the task into three key stages: graph extraction, path
identification, and reasoning. Unlike previous approaches, PoT efficiently
extracts a task-agnostic graph that identifies crucial entities, relations, and
attributes within the problem context. Subsequently, PoT identifies relevant
reasoning chains within the graph corresponding to the posed question,
facilitating inference of potential answers. Experimental evaluations on four
benchmark datasets, demanding long reasoning chains, demonstrate that PoT
surpasses state-of-the-art baselines by a significant margin (maximum 21.3%)
without necessitating fine-tuning or extensive LLM calls. Furthermore, as
opposed to prior neuro-symbolic methods, PoT exhibits improved resilience
against LLM errors by leveraging the compositional nature of graphs.
 | 2024-12-23T20:27:12Z | [Link](http://arxiv.org/abs/2412.17963v1) |
| Other | Contrato360 2.0: A Document and Database-Driven Question-Answer System
  using Large Language Models and Agents |   We present a question-and-answer (Q\&A) application designed to support the
contract management process by leveraging combined information from contract
documents (PDFs) and data retrieved from contract management systems
(database). This data is processed by a large language model (LLM) to provide
precise and relevant answers. The accuracy of these responses is further
enhanced through the use of Retrieval-Augmented Generation (RAG), text-to-SQL
techniques, and agents that dynamically orchestrate the workflow. These
techniques eliminate the need to retrain the language model. Additionally, we
employed Prompt Engineering to fine-tune the focus of responses. Our findings
demonstrate that this multi-agent orchestration and combination of techniques
significantly improve the relevance and accuracy of the answers, offering a
promising direction for future information systems.
 | 2024-12-23T19:54:28Z | [Link](http://arxiv.org/abs/2412.17942v1) |
| Other | VITRO: Vocabulary Inversion for Time-series Representation Optimization |   Although LLMs have demonstrated remarkable capabilities in processing and
generating textual data, their pre-trained vocabularies are ill-suited for
capturing the nuanced temporal dynamics and patterns inherent in time series.
The discrete, symbolic nature of natural language tokens, which these
vocabularies are designed to represent, does not align well with the
continuous, numerical nature of time series data. To address this fundamental
limitation, we propose VITRO. Our method adapts textual inversion optimization
from the vision-language domain in order to learn a new time series per-dataset
vocabulary that bridges the gap between the discrete, semantic nature of
natural language and the continuous, numerical nature of time series data. We
show that learnable time series-specific pseudo-word embeddings represent time
series data better than existing general language model vocabularies, with
VITRO-enhanced methods achieving state-of-the-art performance in long-term
forecasting across most datasets.
 | 2024-12-23T19:24:51Z | [Link](http://arxiv.org/abs/2412.17921v1) |
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
 | 2024-12-23T17:47:53Z | [Link](http://arxiv.org/abs/2412.17743v2) |
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
 | 2024-12-23T16:16:30Z | [Link](http://arxiv.org/abs/2412.17690v2) |
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
| Other | The Power of Adaptation: Boosting In-Context Learning through Adaptive
  Prompting |   Large Language Models (LLMs) have demonstrated exceptional abilities across a
broad range of language-related tasks, including generating solutions to
complex reasoning problems. An effective technique to enhance LLM performance
is in-context learning, which encourages a step-by-step reasoning process by
including explanatory examples to guide the model's responses. However,
selecting appropriate exemplars for the model poses a challenge, as each
dataset demands a distinct set of exemplars to enable the LLM to learn
effectively and perform well on the test set. Current studies often rely on
uncertainty- or diversity-based selection strategies to select exemplars for
annotation and to improve model learning. However, these studies typically
employ a non-adaptive approach, selecting a set of exemplars all at once. We
argue that this non-adaptive strategy may result in a set of exemplars with
high redundancy in terms of the knowledge covered, ultimately reducing their
overall informativeness. To address this limitation, we propose
\textsc{Adaptive-Prompt}, a novel method that adaptively selects exemplars by
leveraging model feedback from previously chosen exemplars. Experimental
results show that \textsc{Adaptive-Prompt} significantly enhances LLM
performance across a variety of reasoning tasks.
 | 2024-12-23T15:49:43Z | [Link](http://arxiv.org/abs/2412.17891v1) |
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
| Finance | Decentralized Intelligence in GameFi: Embodied AI Agents and the
  Convergence of DeFi and Virtual Ecosystems |   In the rapidly evolving landscape of GameFi, a fusion of gaming and
decentralized finance (DeFi), there exists a critical need to enhance player
engagement and economic interaction within gaming ecosystems. Our GameFi
ecosystem aims to fundamentally transform this landscape by integrating
advanced embodied AI agents into GameFi platforms. These AI agents, developed
using cutting-edge large language models (LLMs), such as GPT-4 and Claude AI,
are capable of proactive, adaptive, and contextually rich interactions with
players. By going beyond traditional scripted responses, these agents become
integral participants in the game's narrative and economic systems, directly
influencing player strategies and in-game economies. We address the limitations
of current GameFi platforms, which often lack immersive AI interactions and
mechanisms for community engagement or creator monetization. Through the deep
integration of AI agents with blockchain technology, we establish a
consensus-driven, decentralized GameFi ecosystem. This ecosystem empowers
creators to monetize their contributions and fosters democratic collaboration
among players and creators. Furthermore, by embedding DeFi mechanisms into the
gaming experience, we enhance economic participation and provide new
opportunities for financial interactions within the game. Our approach enhances
player immersion and retention and advances the GameFi ecosystem by bridging
traditional gaming with Web3 technologies. By integrating sophisticated AI and
DeFi elements, we contribute to the development of more engaging, economically
robust, and community-centric gaming environments. This project represents a
significant advancement in the state-of-the-art in GameFi, offering insights
and methodologies that can be applied throughout the gaming industry.
 | 2024-12-24T18:56:00Z | [Link](http://arxiv.org/abs/2412.18601v1) |
| Finance | Combining GPT and Code-Based Similarity Checking for Effective Smart
  Contract Vulnerability Detection |   With the rapid growth of blockchain technology, smart contracts are now
crucial to Decentralized Finance (DeFi) applications. Effective vulnerability
detection is vital for securing these contracts against hackers and enhancing
the accuracy and efficiency of security audits. In this paper, we present
SimilarGPT, a unique vulnerability identification tool for smart contract,
which combines Generative Pretrained Transformer (GPT) models with Code-based
similarity checking methods. The main concept of the SimilarGPT tool is to
measure the similarity between the code under inspection and the secure code
from third-party libraries. To identify potential vulnerabilities, we connect
the semantic understanding capability of large language models (LLMs) with
Code-based similarity checking techniques. We propose optimizing the detection
sequence using topological ordering to enhance logical coherence and reduce
false positives during detection. Through analysis of code reuse patterns in
smart contracts, we compile and process extensive third-party library code to
establish a comprehensive reference codebase. Then, we utilize LLM to conduct
an indepth analysis of similar codes to identify and explain potential
vulnerabilities in the codes. The experimental findings indicate that
SimilarGPT excels in detecting vulnerabilities in smart contracts, particularly
in missed detections and minimizing false positives.
 | 2024-12-24T07:15:48Z | [Link](http://arxiv.org/abs/2412.18225v1) |
| Health | Research on the Proximity Relationships of Psychosomatic Disease
  Knowledge Graph Modules Extracted by Large Language Models |   As social changes accelerate, the incidence of psychosomatic disorders has
significantly increased, becoming a major challenge in global health issues.
This necessitates an innovative knowledge system and analytical methods to aid
in diagnosis and treatment. Here, we establish the ontology model and entity
types, using the BERT model and LoRA-tuned LLM for named entity recognition,
constructing the knowledge graph with 9668 triples. Next, by analyzing the
network distances between disease, symptom, and drug modules, it was found that
closer network distances among diseases can predict greater similarities in
their clinical manifestations, treatment approaches, and psychological
mechanisms, and closer distances between symptoms indicate that they are more
likely to co-occur. Lastly, by comparing the proximity d and proximity z score,
it was shown that symptom-disease pairs in primary diagnostic relationships
have a stronger association and are of higher referential value than those in
diagnostic relationships. The research results revealed the potential
connections between diseases, co-occurring symptoms, and similarities in
treatment strategies, providing new perspectives for the diagnosis and
treatment of psychosomatic disorders and valuable information for future mental
health research and practice.
 | 2024-12-24T13:24:01Z | [Link](http://arxiv.org/abs/2412.18419v1) |
| Health | Real-world Deployment and Evaluation of PErioperative AI CHatbot (PEACH)
  -- a Large Language Model Chatbot for Perioperative Medicine |   Large Language Models (LLMs) are emerging as powerful tools in healthcare,
particularly for complex, domain-specific tasks. This study describes the
development and evaluation of the PErioperative AI CHatbot (PEACH), a secure
LLM-based system integrated with local perioperative guidelines to support
preoperative clinical decision-making. PEACH was embedded with 35 institutional
perioperative protocols in the secure Claude 3.5 Sonet LLM framework within
Pair Chat (developed by Singapore Government) and tested in a silent deployment
with real-world data. Accuracy, safety, and usability were assessed. Deviations
and hallucinations were categorized based on potential harm, and user feedback
was evaluated using the Technology Acceptance Model (TAM). Updates were made
after the initial silent deployment to amend one protocol.
  In 240 real-world clinical iterations, PEACH achieved a first-generation
accuracy of 97.5% (78/80) and an overall accuracy of 96.7% (232/240) across
three iterations. The updated PEACH demonstrated improved accuracy of 97.9%
(235/240), with a statistically significant difference from the null hypothesis
of 95% accuracy (p = 0.018, 95% CI: 0.952-0.991). Minimal hallucinations and
deviations were observed (both 1/240 and 2/240, respectively). Clinicians
reported that PEACH expedited decisions in 95% of cases, and inter-rater
reliability ranged from kappa 0.772-0.893 within PEACH and 0.610-0.784 among
attendings.
  PEACH is an accurate, adaptable tool that enhances consistency and efficiency
in perioperative decision-making. Future research should explore its
scalability across specialties and its impact on clinical outcomes.
 | 2024-12-24T02:14:13Z | [Link](http://arxiv.org/abs/2412.18096v1) |
| Health | CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language
  Models |   Causal reasoning capabilities are essential for large language models (LLMs)
in a wide range of applications, such as education and healthcare. But there is
still a lack of benchmarks for a better understanding of such capabilities.
Current LLM benchmarks are mainly based on conversational tasks, academic math
tests, and coding tests. Such benchmarks evaluate LLMs in well-regularized
settings, but they are limited in assessing the skills and abilities to solve
real-world problems. In this work, we provide a benchmark, named by CARL-GT,
which evaluates CAusal Reasoning capabilities of large Language models using
Graphs and Tabular data. The benchmark has a diverse range of tasks for
evaluating LLMs from causal graph reasoning, knowledge discovery, and
decision-making aspects. In addition, effective zero-shot learning prompts are
developed for the tasks. In our experiments, we leverage the benchmark for
evaluating open-source LLMs and provide a detailed comparison of LLMs for
causal reasoning abilities. We found that LLMs are still weak in casual
reasoning, especially with tabular data to discover new insights. Furthermore,
we investigate and discuss the relationships of different benchmark tasks by
analyzing the performance of LLMs. The experimental results show that LLMs have
different strength over different tasks and that their performance on tasks in
different categories, i.e., causal graph reasoning, knowledge discovery, and
decision-making, shows stronger correlation than tasks in the same category.
 | 2024-12-23T20:34:32Z | [Link](http://arxiv.org/abs/2412.17970v1) |
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
| Education | Molly: Making Large Language Model Agents Solve Python Problem More
  Logically |   Applying large language models (LLMs) as teaching assists has attracted much
attention as an integral part of intelligent education, particularly in
computing courses. To reduce the gap between the LLMs and the computer
programming education expert, fine-tuning and retrieval augmented generation
(RAG) are the two mainstream methods in existing researches. However,
fine-tuning for specific tasks is resource-intensive and may diminish the
model`s generalization capabilities. RAG can perform well on reducing the
illusion of LLMs, but the generation of irrelevant factual content during
reasoning can cause significant confusion for learners. To address these
problems, we introduce the Molly agent, focusing on solving the proposed
problem encountered by learners when learning Python programming language. Our
agent automatically parse the learners' questioning intent through a
scenario-based interaction, enabling precise retrieval of relevant documents
from the constructed knowledge base. At generation stage, the agent reflect on
the generated responses to ensure that they not only align with factual content
but also effectively answer the user's queries. Extensive experimentation on a
constructed Chinese Python QA dataset shows the effectiveness of the Molly
agent, indicating an enhancement in its performance for providing useful
responses to Python questions.
 | 2024-12-24T02:08:38Z | [Link](http://arxiv.org/abs/2412.18093v1) |
| Education | LLM-Driven Feedback for Enhancing Conceptual Design Learning in Database
  Systems Courses |   The integration of LLM-generated feedback into educational settings has shown
promise in enhancing student learning outcomes. This paper presents a novel
LLM-driven system that provides targeted feedback for conceptual designs in a
Database Systems course. The system converts student-created
entity-relationship diagrams (ERDs) into JSON format, allows the student to
prune the diagram by isolating a relationship, extracts relevant requirements
for the selected relationship, and utilizes a large language model (LLM) to
generate detailed feedback. Additionally, the system creates a tailored set of
questions and answers to further aid student understanding. Our pilot
implementation in a Database System course demonstrates effective feedback
generation that helped the students improve their design skills.
 | 2024-12-23T18:39:11Z | [Link](http://arxiv.org/abs/2412.17892v1) |
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
