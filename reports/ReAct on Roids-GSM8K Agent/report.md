# ReAct on Roids-GSM8K Agent: Training Script and Results

## Overview

This report explains the concept behind the "ReAct on Roids" agent, details the training script used, and summarizes the results of the experiment.

## What is ReAct on Roids?

"ReAct on Roids" builds on the ReAct prompting method, which is commonly used for AI agents (e.g., with OpenAI or Anthropic APIs). In ReAct, agents are prompted to alternate between generating their thoughts and actions/tool calls in multiple steps until the task is solved. Our approach combines this ReAct prompting with reinforcement learning, encouraging the agent to actively strategize using a think/act loop—hence the name "on Roids."

### The Think/Act Loop

The agent is prompted to structure its reasoning and actions using the following tags:
- `<think>...</think>` for internal reasoning
- `<code>...</code>` for executable Python code
- `<output>...</output>` for code execution results
- `<answer>...</answer>` for the final answer

This format encourages step-by-step problem solving, with the final answer always enclosed in `<answer>` tags.

## Training Script

You can find the full training script [here](gsm8k_agent_react_on_roids.py)

### System Prompt

The system prompt instructs the model to use the code interpreter and respond in the ReAct format, providing two worked examples to guide the agent.

### Custom Trainer

A custom `GRPOTrainerWithEval` class was implemented to evaluate the agent every N steps on a subset of the GSM8K test set. This evaluation is optional but useful for tracking progress.

### Reward Functions

Three reward functions were used:
- **`integer_reward_func`**: Rewards the agent for producing integer answers in `<answer>` tags (since GSM8K answers are always integers).
- **`correctness_and_strategy_reward_func`**: Rewards correct answers, with an additional bonus for shorter (more efficient) completions. Note: The model is not rewarded for brevity alone, but for being both correct and concise.
- **`react_reward_func`**: Rewards adherence to the ReAct format, ensuring the agent uses the code interpreter at least once per response.

### Tokenizer Issue and Solution

A challenge arose because the code interpreter output tags (`<output>...</output>`) were not reliably tokenized by the default tokenizer. To fix this, the tokens `<output>` and `</output>` were added to the model's tokenizer, the embedding layer was resized, and the updated model was uploaded to the Hugging Face Hub. This custom model (`Qwen2.5-0.5B-Instruct-with-output-tokens`) was used instead of the standard `Qwen-0.5B`.

### Data Preparation and Environment

The script also handles preprocessing of training and evaluation data, loading the tokenizer, and initializing the code execution environment.

## Running the Script

The experiment was run on a 4x3090 GPU instance. The vLLM server was started first:

```bash
CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model "August4293/Qwen2.5-0.5B-Instruct-with-output-tokens" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --dtype "bfloat16" \
    --max_model_len 2048 &
```
> The `&` might be necessary to run the server in the background, as the terminal may freeze otherwise.

Then, the training script was launched:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_config.yml gsm8k_agent_react_on_roids.py
```

## Results

The agent was trained on fewer than 1,000 GSM8K questions (training was stopped early for resource reasons). The results were promising:  
- The agent learned to use the ReAct format and invoked code execution in about 90% of responses.
- Correct answers were typically concise and well-formatted.

You can view the full run and generated responses on [Weights & Biases](https://wandb.ai/moh-murr/ReAct_on_Roids/runs/26o633zo?nw=nwusermohmurr).

### Generated Responses in the last step

**Q:** Joshua and Justin shared $40. Joshua’s share was thrice as much as Justin’s. How much is Joshua's share?
```
<think>Let's denote Justin's share as \( J \) and Joshua's share as \( O \). We know from the problem that \( J + O = 40 \) (the total amount shared) and that \( O = 3J \) (Joshua's share is three times what Justin has). We can substitute \( O \) in the first equation with \( 3J \). This gives us \( J + 3J = 40 \).</think><code>print((40 * 3) / 4)</code><output>Execution(Results: [], Logs: Logs(stdout: ['30.0\n'], stderr: []), Error: None)</output><answer>30</answer>
```

---

**Q:**  Steven has 4 times as many shirts as Andrew. Andrew has 6 times as many shirts as Brian. If Brian has 3 shirts, how many does Steven have?

```
<think>Given that Andrew has 6 times as many shirts as Brian, and Brian has 3 shirts, we can determine the number of shirts Andrew has by \(6 \times 3\). Then, we determine the number of shirts Steven has by \(4 \times\) the number of shirts Andrew has, since Steven has 4 times as many shirts as Andrew.</think><code>print(4 * 6 * 3)</code><output>Execution(Results: [], Logs: Logs(stdout: ['72\n'], stderr: []), Error: None)</output><answer>72</answer>
```

---

**Q:**   At the Greene Nursery, there are 1491 red roses, 3025 yellow carnations, and 1768 white roses. How many flowers are at the Greene Nursery in total?

```
<think>First calculate the total number of flowers by adding the number of red roses, yellow carnations, and white roses together.</think><code>print(1491 + 3025 + 1768)</code><output>Execution(Results: [], Logs: Logs(stdout: ['6284\n'], stderr: []), Error: None)</output><answer>6284</answer
```

---

**Q:** Beth had a set of 125 gold coins.  Carl gave Beth a gift of an additional 35 gold coins.  Then, Beth decided to sell half of her coins.  How many coins did Beth sell?

```
<think>First, we need to find out how many coins Beth had after Carl gives her more coins.</think><code>print(125 + 35)</code><output>Execution(Results: [], Logs: Logs(stdout: ['160\n'], stderr: []), Error: None)</output><think>The equation above calculates the total number of coins Beth had after receiving Carl's gift, but it doesn't directly tell us the number of coins Beth sold.</think><code>print(160 / 2)</code><output>Execution(Results: [], Logs: Logs(stdout: ['80.0\n'], stderr: []), Error: None)</output><answer>80</answer>
```

---

**Q:**  Tony tries to drink more water, so he has been tracking how much water he drinks. Yesterday, he drank 48 ounces of water, which is 4% less than what he drank two days ago. How much water did Tony drink two days ago?

```
<think>The water Tony drank yesterday is 40% less than what he drank two days ago, so the difference between what he drank last two days and the water he drank yesterday is 4% of that difference.</think><code>print(48 / (1 - 0.04))</code><output>Execution(Results: [], Logs: Logs(stdout: ['50.0\n'], stderr: []), Error: None)</output><answer>50.0</answer>
```

---