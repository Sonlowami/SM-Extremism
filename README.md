# 04-637 Mobile Big Data Analytics and Management

> This repository contains the implementation and analysis for **Assignment 3: Extremism Analysis, Prompt Engineering, and Counter-Narrative Generation**. The project builds on prior extremism detection work by introducing a multi-dimensional classification framework and evaluating large language models (LLMs) in realistic content moderation scenarios.

---

## Repository Structure

* `/Part1`: Contains code and analysis for Part 1 (Multi-Dimensional Extremism Classification) - Assignment 2.

* `MBD_Assignment_3.ipynb`: Main notebook containing all experiments, implementation, and analysis.
* `README.md`: Detailed report explaining methodology, design choices, and findings.
* `ve_nve_classifications.csv`: Output dataset with VE/NVE classification, subtype labels, and target group detection.

---

## Instructions to Run the Notebook

This project is designed to run on **Google Colab** due to dependencies on APIs and compute requirements.

1. Open the notebook in Google Colab (or upload manually).
2. Configure required API credentials:

   * Kaggle Username and API Key
   * CMU API Gateway Key (for LLM access)
3. Run all cells sequentially:

   * Dataset download and preprocessing
   * Part A: Multi-dimensional classification
   * Part B: Prompt engineering experiments
   * Part C: Counter-narrative generation

---

# Analysis and Findings

## Part A: Multi-Dimensional Extremism Classification

### Overview

In this section, we extend a binary extremism classifier into a **multi-dimensional system** that captures:

* Whether content is **Violent (VE)** or **Non-Violent Extremist (NVE)**
* The **subtype of violent extremism**
* The **targeted demographic group**
* A **justification grounded in text spans**

We used an LLM-based classification approach.

---

### Definitions and Conceptual Grounding

* **According to the authors, what are the differences between Violent extremism or non-violent extremism?**
Violent extremism must constitute a violent action, or a threat of so. Non-violent extremism does not constitute a threat of violence nor violence itself.
Violent extremists tend to have a low self-esteem compared to NVEs.
Violent extemists also were found by the authors to have been exposed to extreme violence including through the internet or bullying. NVEs do not share the same experience.

* **Based on the paper, how is radicalization and extremism defined?**
The authors argue that radicalization, which involve having different opinions from the mainstream is one of the pathways to extremism-linked violence. However, they note that the majority of people with radical ideas are not involved in violence, and that radicalization is not the only way to extremism.

* **How is violent and non-violent extremism defined?**
The authors describe violent extremism as acts or propositions that, if inacted, would result in murder, attempted murder, manslaughter or serious injury to another person or people. This definition expands to involve people who conduct non-violent behaviors that support violence. On the other hand, the authors left their definition of non-violent extremism to user inference, in which we identified as expression of ideas that diverge from mainstream but do not constitute a threat of action or the action itself. The individuals they collected data from were all convicted of criminal charges linked to extremism of some kind.

---

### Prompt Design Choices

Designing the classification prompt required several key decisions:

* Using **strict academic definitions** to reduce ambiguity between VE and NVE
* Structuring outputs as **JSON** for downstream processing
* Including **multiple tasks in a single prompt** (classification, subtype, target detection)
* Adding **justification requirements** to improve interpretability

A major challenge was balancing **precision vs safety constraints**, as some prompts triggered content filtering.

---

### ⚠️ Model Constraints and Adjustments

While the assignment specified particular models (e.g., GPT-4o-mini), we encountered **frequent safety filtering issues** due to the sensitive nature of extremist content.

* Many inputs were **blocked or refused** by certain models
* This interrupted batch processing and reduced dataset coverage

**To address this:**

* We **switched to alternative models** (e.g., GPT-4.1-mini via CMU gateway) that allowed more stable processing
* We **rephrased prompts** to avoid reproducing harmful content
* We enforced **strict output formatting** to prevent hallucinations or refusals

This adjustment ensured:

* Completion of experiments
* Consistent outputs across all samples
* Reduced API interruptions

---

### Results

* **Most frequent VE subtype:**
  → **Ideological extremism**

* **VE vs NVE distribution:**
  → Slightly more **Non-Violent Extremism (NVE)** than VE

* **Top targeted groups:**

  * Ethnic/Racial groups
  * Nationality groups
  * Religious groups
  * Gender groups
  * Unclear/implicit targets

---

### Key Observations

* Extremist content is often **identity-driven**, targeting specific demographic groups
* A significant portion of content is **non-violent but ideologically extreme**, which poses challenges for moderation systems
* Target identification is sometimes **ambiguous**, especially in sarcastic or indirect language

---

## Part B: Prompt Engineering Comparison

### Overview

This section evaluates how **prompting strategies** affect LLM performance in severity classification.

We compare:

* **Zero-shot prompting**: Directly asking for classification without additional context
* **Chain-of-Thought (CoT) prompting**: Encouraging the model to reason step-by-step before giving a final answer

Across:

* Default temperature
* Temperature = 0.4

---

### Experimental Setup

* Model: **LLaMA 3 (11B Instruct)** *(primary choice as instructed)*
* Dataset: 100 balanced samples (50 VE, 50 NVE)

---

### Severity Levels

* 0 -> Absolute Refusal: No extremist content; clear rejection of violence or radical ideology
* 1 -> Mild Concern: Vague or ambiguous language; no clear extremist intent
* 2 -> Ideological Support: Passive sympathy or endorsement of extremist views (no explicit violence)
* 3 -> Explicit Propaganda: Active promotion or justification of extremist ideology or violence
* 4 -> Operational Extremism: Recruitment, logistics, attack planning, or incitement to imminent violence

---

### Results

#### Consistency Across Runs

* Zero vs CoT (default): **59%**
* Zero vs CoT (temp 0.4): **55%**
* Zero default vs temp 0.4: **84%**
* CoT default vs temp 0.4: **84%**

**Insight:**

* Prompt style has a **larger impact** than temperature
* CoT does not guarantee consistent classification outcomes

---

#### Explanation Length (Verbosity)

* Zero Default: 19.86 words on average
* CoT Default: 58.78 words on average
* Zero Temp 0.4: 19.94 words on average
* CoT Temp 0.4: 61.29 words on average


**Insight:**

* CoT generates significantly more detailed explanations
* Temperature has minimal effect on explanation length


## Part C: Counter-Narrative Generation

### Overview

This section explores the use of **Large Language Models (LLMs) to generate counter-narratives** in response to extremist content. The goal is not only to produce safe responses, but to evaluate how **model choice, prompting strategy, and persona design** influence:

* **Effectiveness**
* **Clarity**
* **Readability**
* **Verbosity**

We frame this as a **real-world content moderation task**, where responses must be persuasive, safe, and understandable to a general audience.

Note: Due to the sensitive nature of the task, we encountered significant **safety filtering and refusal issues** across models, which influenced our experimental design and model selection. `gpt-4.1-mini` was replaced with `gpt-5.4-mini` for better consistency in generation.

---

### Methodology

#### Persona-Based Generation

For each extremist input, we generated **four counter-narratives** using different personas:

1. **Vanilla (Baseline)**

   * Neutral response without a defined perspective
   * Serves as a control condition

2. **Educator**

   * Focus: Correct misinformation using facts and logic
   * Tone: Calm, structured, and informative

3. **Compassionate NGO Worker**

   * Focus: Appeal to empathy and shared humanity
   * Tone: Emotional, human-centered

4. **Law Enforcement Officer**

   * Focus: Deter harmful behavior through consequences
   * Tone: Authoritative and cautionary

All generations were performed at **temperature = 0.7** to encourage diversity while maintaining coherence.

---

### Model Comparisons

We evaluated models across two axes:

#### 1. Training Paradigm

* **GPT-4.1-mini (RLHF-based) -> switched to GPT-5.4-mini**
* **LLaMA 3 (Instruction-tuned)**

#### 2. Cost vs Quality Trade-off

* **Claude Sonnet (higher cost, higher capability)**
* **Claude Haiku (lower cost, faster, cheaper)**

---

### Evaluation Metrics

We evaluated outputs using both **quantitative** and **LLM-based qualitative metrics**:

#### 1. Verbosity

* Measured as **word count**
* Used to compare how detailed each model/persona is

#### 2. Readability (Flesch Reading Ease)

Converted to a 1–5 scale:

* 5 → Very easy (≥80)
* 4 → Easy (≥70)
* 3 → Standard (≥60)
* 2 → Difficult (≥50)
* 1 → Very difficult (<50)

#### 3. LLM-Based Clarity Score

* Evaluated using **GPT-4o-mini**
* Target audience: **General public (≈ 8th grade level)**
* Scale:

  * 1 = Very unclear
  * 5 = Very clear

---

### Results and Analysis

#### 1. Verbosity vs Model Behavior

We generated a total of **2400 counter-narratives** across all models and personas. The verbosity analysis reveals clear differences in how models construct responses:

| Model         | Avg Words | Median | Std Dev | Min | Max |
| ------------- | --------- | ------ | ------- | --- | --- |
| Claude Haiku  | 176.29    | 176.5  | 43.95   | 86  | 309 |
| Claude Sonnet | 174.28    | 177.0  | 21.03   | 98  | 225 |
| GPT-5-mini    | 100.70    | 98.0   | 29.38   | 46  | 208 |
| LLaMA 3 (11B) | 186.69    | 252.0  | 155.17  | 5   | 415 |

* **Most verbose model:** LLaMA 3 (avg ~187 words)
* **Least verbose model:** GPT-5-mini (avg ~101 words)

---

#### Key Observations

* **LLaMA 3 shows extremely high variance** (std = 155), indicating unstable generation behavior:

  * Some outputs are extremely short (5 words)
  * Others are excessively long (415 words)
  * Median (252) >> mean (186), suggesting **frequent long outliers**

* **Claude Sonnet is the most consistent model**:

  * Lowest standard deviation (21)
  * Tight range of outputs
  * Predictable verbosity across prompts

* **Claude Haiku behaves similarly to Sonnet but with slightly more spread**, suggesting a trade-off between cost and stability.

* **GPT-5-mini is significantly more concise**:

  * Produces ~40% shorter responses than other models
  * Likely optimized for efficiency and cost

**Key Insight:**
Verbosity is not just about length—it reflects **model stability and control**.

* High variance (LLaMA 3) → less reliable outputs
* Low variance (Claude Sonnet) → more production-ready behavior

---

#### 2. Persona Impact on Output Quality

We further analyzed verbosity across personas:

| Model         | Educator | Law Enforcement | NGO    | Vanilla |
| ------------- | -------- | --------------- | ------ | ------- |
| Claude Haiku  | 182.91   | 180.19          | 196.03 | 146.04  |
| Claude Sonnet | 177.43   | 174.23          | 177.16 | 168.32  |
| GPT-5-mini    | 114.53   | 98.27           | 102.41 | 87.61   |
| LLaMA 3 (11B) | 166.94   | 217.89          | 151.76 | 210.17  |

---

#### Persona-Specific Insights

* **NGO persona consistently produces the longest responses**

  * Highest verbosity across most models (e.g., 196 words in Claude Haiku)
  * Reflects emphasis on **empathy, storytelling, and emotional appeal**

* **Law enforcement persona is highly variable**

  * Especially in LLaMA 3 (217 words), suggesting **over-generation or repetition**
  * Likely due to authoritative tone expanding into warnings and consequences

* **Educator persona is balanced and consistent**

  * Moderate verbosity across all models
  * Suggests structured explanations without excessive elaboration

* **Vanilla baseline produces the shortest responses**

  * Particularly in GPT-5-mini (87 words)
  * Indicates lack of strong guidance leads to **minimal, less persuasive outputs**

---

#### Cross-Model Behavioral Patterns

* **GPT-5-mini remains concise across all personas**

  * Even when prompted emotionally (NGO), it avoids long outputs
  * Suggests strong internal constraints or optimization for brevity

* **LLaMA 3 is highly sensitive to persona**

  * Large swings between personas (151 → 217 words)
  * Indicates weaker control over stylistic constraints

* **Claude models are robust to persona changes**

  * Maintain relatively stable verbosity across roles
  * Better suited for consistent deployment

---

**Key Insight:**
Persona does not just change *tone*—it significantly affects **response length, structure, and consistency**.

* NGO → longer, emotional, narrative-heavy
* Educator → structured, moderate length
* Law enforcement → directive, sometimes verbose
* Vanilla → shortest, least effective

---

#### 3. Readability vs Clarity

To evaluate counter-narrative quality, we compared:

* **Traditional readability metrics** (Flesch Reading Ease, Flesch-Kincaid Grade)
* **LLM-based clarity scores** (GPT-4.1-mini, 1–5 scale)

---

### Readability Results (by Model)

| Model         | Flesch Reading Ease | FK Grade | Flesch (1–5) |
| ------------- | ------------------- | -------- | ------------ |
| Claude Haiku  | 29.02               | 13.79    | 1.01         |
| Claude Sonnet | -14.40              | 24.97    | 1.00         |
| GPT-5-mini    | 34.88               | 12.86    | 1.12         |
| LLaMA 3 (11B) | 42.11               | 11.01    | 1.90         |

---

### Key Observations

* **All models score poorly on traditional readability**

  * Most outputs fall in **college-level difficulty (scale ≈ 1–2)**
  * Even the *best* model (LLaMA 3) only reaches **1.90 / 5**

* **Claude Sonnet performs worst in readability**

  * Negative Flesch score (-14.4)
  * Very high grade level (~25)
  * Indicates **extremely complex sentence structures**

* **GPT-5-mini produces slightly more readable outputs**

  * Lower grade level (~12.9)
  * Simpler and shorter responses

* **LLaMA 3 achieves the highest readability score (1.90)**

  * Despite being the most verbose model
  * Suggests that **verbosity does not necessarily reduce readability**

---

### Readability by Persona

| Model         | Educator | Law Enforcement | NGO  | Vanilla |
| ------------- | -------- | --------------- | ---- | ------- |
| Claude Haiku  | 1.00     | 1.00            | 1.05 | 1.00    |
| Claude Sonnet | 1.00     | 1.00            | 1.00 | 1.00    |
| GPT-5-mini    | 1.07     | 1.08            | 1.22 | 1.10    |
| LLaMA 3 (11B) | 2.03     | 1.63            | 2.27 | 1.68    |

---

### Persona-Level Insights

* **NGO persona consistently improves readability**

  * Highest scores across models (e.g., 2.27 for LLaMA 3)
  * Likely due to **simpler, more conversational language**

* **Educator and Law Enforcement personas reduce readability**

  * Use of formal, technical, or authoritative language
  * Leads to **longer sentences and higher complexity**

* **Claude models are insensitive to persona changes**

  * Nearly constant readability (1.0)
  * Suggests rigid generation style

* **LLaMA 3 is highly responsive to persona**

  * Large variation (1.63 → 2.27)
  * Indicates stronger stylistic adaptability

---

### LLM-Based Clarity Scores

| Model         | Avg. Clarity Score (1–5) |
| ------------- | ------------------------ |
| Claude Haiku  | 4.02                     |
| Claude Sonnet | 4.23                     |
| GPT-5-mini    | 4.87                     |
| LLaMA 3 (11B) | 4.54                     |

#### Clarity by Persona

| Model         | Educator | Law Enforcement | NGO  | Vanilla |
| ------------- | -------- | --------------- | ---- | ------- |
| Claude Haiku  | 3.73     | 4.01            | 4.01 | 4.31    |
| Claude Sonnet | 4.15     | 4.25            | 4.31 | 4.22    |
| GPT-5-mini    | 4.78     | 4.90            | 4.89 | 4.89    |
| LLaMA 3 (11B) | 4.51     | 4.51            | 4.61 | 4.51    |

---

### Key Observations on Clarity

* **GPT-5-mini is the clearest model overall**

  * Highest average clarity (4.87 / 5)
  * Consistently strong across all personas (4.78–4.90)

* **Persona influences clarity less than readability**

  * Claude models show minor improvements with vanilla or NGO personas
  * LLaMA 3 clarity is stable across personas (~4.5–4.6)

* **Clarity does not perfectly track readability**

  * LLaMA 3 has higher readability than GPT-5-mini for some personas (e.g., NGO)
  * But GPT-5-mini scores higher in clarity despite lower Flesch scores
  * Suggests **clarity captures comprehension beyond surface-level sentence simplicity**

* **Claude Sonnet**

  * Lowest readability, yet clarity remains decent (~4.23 overall)
  * Indicates that complex language can still be clear in meaning

---

### Disagreement Between Readability and Clarity

* **Traditional readability metrics** focus on **sentence length and word difficulty**
* **LLM clarity scores** reflect **how understandable and logically coherent text is**
* Result:

  * A model like LLaMA 3 may be more “readable” superficially, but GPT-5-mini produces **easier-to-understand counter-narratives**
  * Persona-driven readability improvements (NGO) do not always translate to higher clarity

### Top 5 Examples (LLM Clarity and Flesch Scores Disagree)

--- Example 1 ---
Model: gpt-5-mini, Persona: vanilla

Counter-narrative: I can help with a constructive counter-narrative:

It’s important to focus on facts and respectful language instead of insults or dehumanizing labels. Political disagreements should be addressed throu...

LLM Clarity Score: 5

Flesch Scale (1-5): 1

Flesch Reading Ease: 24.0

Difference: 4

--- Example 2 ---
Model: gpt-5-mini, Persona: educator

Counter-narrative: It’s understandable to feel frustrated about politics, but comparing people to Nazis or reducing complex human beings to slurs only inflames conflict and shuts down real discussion. A more productive ...

LLM Clarity Score: 5

Flesch Scale (1-5): 1

Flesch Reading Ease: 26.7

Difference: 4

--- Example 3 ---
Model: gpt-5-mini, Persona: ngo

Counter-narrative: We can disagree strongly about immigration or politics without using dehumanizing labels or hate. Calling people “Nazis” or “illegal aliens” shuts down understanding and makes it harder to solve real ...

LLM Clarity Score: 5

Flesch Scale (1-5): 1

Flesch Reading Ease: 35.4

Difference: 4

--- Example 4 ---
Model: gpt-5-mini, Persona: law_enforcement

Counter-narrative: Respectful disagreement matters. Using insults or dehumanizing labels only deepens division and distracts from real issues. People can debate immigration policy, elections, and public safety without a...

LLM Clarity Score: 5

Flesch Scale (1-5): 1

Flesch Reading Ease: 12.1

Difference: 4

--- Example 5 ---
Model: gpt-5-mini, Persona: vanilla

Counter-narrative: Violence and terrorism are never justified, and extremist groups hurt the very communities they claim to defend. Real strength comes from protecting human life, respecting differences, and solving con...

LLM Clarity Score: 5

Flesch Scale (1-5): 1

Flesch Reading Ease: 46.6

Difference: 4

### Why Metrics Disagree?

The disagreement arises because the two metrics measure fundamentally different things:

| Metric             | What it Measures                     | Limitation              |
| ------------------ | ------------------------------------ | ----------------------- |
| Flesch Readability | Sentence length & word complexity    | Ignores meaning         |
| LLM Clarity Score  | Coherence, usefulness, understanding | Subjective but holistic |


## Correlation

### 1. Verbosity vs Readability and Clarity

| Metric Pair                       | Correlation | Interpretation                                                                             |
| --------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| Word count vs Flesch Reading Ease | -0.362      | Longer texts are moderately harder to read (surface-level readability decreases)           |
| Word count vs LLM Clarity Score   | -0.526      | Longer texts tend to slightly reduce perceived clarity, but less than readability predicts |

**Observation:**

* Traditional readability penalizes longer outputs more gently than clarity, but clarity also declines with extreme verbosity.
* Shorter, more concise counter-narratives (e.g., GPT-5-mini) achieve **higher clarity and better readability** simultaneously.

---

### 2. Model-Level Summary: Verbosity, Readability, Clarity

| Model         | Avg Words | Avg Flesch RE | Avg Flesch (1–5) | Avg LLM Clarity |
| ------------- | --------- | ------------- | ---------------- | --------------- |
| Claude Haiku  | 176.3     | 29.02         | 1.01             | 4.02            |
| Claude Sonnet | 174.3     | -14.40        | 1.00             | 4.23            |
| GPT-5-mini    | 100.7     | 34.88         | 1.12             | 4.87            |
| LLaMA 3 (11B) | 186.7     | 42.11         | 1.90             | 4.54            |

**Insights:**

* **GPT-5-mini** is the shortest output (≈100 words) but has the **highest clarity** (4.87).
* **Claude Sonnet and Haiku** are verbose (~175 words), low readability, moderate clarity.
* **LLaMA 3** is the most verbose (~187 words) with highest Flesch score among all models (1.90), but clarity slightly lower than GPT-5-mini (4.54).

---

### 3. Model Comparison: GPT-5-mini vs Claude Sonnet (Task 3)

| Model         | Avg Words | Avg Flesch RE | Avg Flesch (1–5) | Avg LLM Clarity |
| ------------- | --------- | ------------- | ---------------- | --------------- |
| Claude Sonnet | 174.3     | -14.40        | 1.00             | 4.23            |
| GPT-5-mini    | 100.7     | 34.88         | 1.12             | 4.87            |

**Takeaways:**

* GPT-5-mini produces **shorter, more readable, and clearer counter-narratives** than Claude Sonnet.
* Despite LLaMA 3 having slightly higher traditional readability, GPT-5-mini is **overall the best model** for clarity-focused counter-narratives.