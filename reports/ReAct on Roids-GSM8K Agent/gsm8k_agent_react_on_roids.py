import re
import os
from trl import GRPOConfig, GRPOTrainer, CodeAgentEnvironment, E2BExecuter, prepare_data_for_e2b_agent,VLLMClientGenerationConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import wandb


def extract_ground_truth(example):
    answer_str = example["answer"]
    final_answer_str = answer_str.split("#### ")[-1]
    final_answer_int = int(final_answer_str.replace(",", "")) # Remove commas for thousands separators
    return {
        "ground_truth": final_answer_int,
        "prompt": example["question"]
    }

system_prompt = (
    "You are a math assistant that solves problems step-by-step using Python code.\n\n"
    "FORMATTING RULES:\n"
    "1. Put your thinking process inside <think>...</think> tags\n"
    "2. Put executable Python code inside <code>...</code> tags\n"
    "3. The code execution result will appear in <output>...</output> tags\n"
    "4. Put your final numerical answer inside <answer>...</answer> tags\n"
    "5. You can use multiple think-code-output steps for complex problems\n"
    "6. Always show your complete reasoning and finish with one final answer\n\n"
    "EXAMPLES:\n\n"
    
    "Example 1:\n"
    "user: If a shirt costs $25 and is on sale for 30% off, what is the sale price?\n"
    "assistant: <think>I need to find what 30% of $25 is, then subtract that from $25.</think>"
    "<code>print(25 * 0.3)</code>"
    "<output>7.5</output>"
    "<think>Now I subtract the discount from the original price.</think>"
    "<code>print(25 - 7.5)</code>"
    "<output>17.5</output>"
    "<answer>17.5</answer>"
    
    "Example 2:\n"
    "user: John has 5 boxes of pencils with 12 pencils in each box. He gives 7 pencils to his friend. How many pencils does John have left?\n"
    "assistant: <think>First, I'll calculate the total number of pencils John started with by multiplying the number of boxes by the number of pencils per box.</think>"
    "<code>print(5 * 12)</code>"
    "<output>60</output>"
    "<think>Now I'll subtract the number of pencils John gave to his friend.</think>"
    "<code>print(60 - 7) </code>"
    "<output>53</output>"
    "<answer>53</answer>"
)
class GRPOTrainerWithEval(GRPOTrainer):
    def __init__(
        self,
        *args,
        eval_dataset=None,
        eval_gen_config=None,
        eval_every_n_steps=20,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_dataset = eval_dataset
        self.eval_gen_config = eval_gen_config
        self.eval_every_n_steps = eval_every_n_steps
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)

        if self.accelerator.is_main_process and self.eval_dataset and self.state.global_step > 0 and \
           (self.state.global_step % self.eval_every_n_steps == 0):

            if not (self.eval_dataset.column_names and \
                    'prompt' in self.eval_dataset.column_names and \
                    'ground_truth' in self.eval_dataset.column_names):
                return result

            eval_prompts_list = self.eval_dataset['prompt']
            ground_truths = self.eval_dataset['ground_truth']

            if not eval_prompts_list:
                print("[EvalHook] Skipping evaluation: eval_dataset['prompt'] is empty.")
                return result
            
            if len(eval_prompts_list) != len(ground_truths):
                print(f"[EvalHook] Skipping evaluation: Mismatch between number of prompts ({len(eval_prompts_list)}) and ground_truths ({len(ground_truths)}).")
                return result

            if self.eval_gen_config is None:
                print("[EvalHook] Warning: eval_gen_config is None. Using default generation parameters from environment, which might not be suitable (e.g., n > 1).")
            
            responses = self.environment.run_agent(
                vllm_client=self.vllm_client,
                generation_config=self.eval_gen_config,
                prompts=eval_prompts_list,
            )

            if not responses or len(responses) != len(eval_prompts_list):
                print(f"[EvalHook] Skipping evaluation: Number of responses ({len(responses)}) "
                      f"does not match number of prompts ({len(eval_prompts_list)}).")
                return result

            correct_answers = 0
            total_prompts_evaluated = len(responses)

            for i, full_response_text in enumerate(responses):
                original_prompt_text = eval_prompts_list[i]
                completion_text = ""

                if full_response_text.startswith(original_prompt_text):
                    completion_text = full_response_text[len(original_prompt_text):]
                else:
                    completion_text = full_response_text
                    
                match = self.answer_pattern.search(completion_text)
                if match:
                    answer_content = match.group(1).strip().replace(",", "")
                    try:
                        extracted_answer_int = int(answer_content)
                        if extracted_answer_int == int(ground_truths[i]):
                            correct_answers += 1
                    except ValueError:
                        pass 
                    except TypeError:
                        pass
            
            success_rate = 0.0
            if total_prompts_evaluated > 0:
                success_rate = float(correct_answers) / total_prompts_evaluated
            
            log_metric_name = "gsm300"
            print(f"[EvalHook] {log_metric_name}: {success_rate:.4f} ({correct_answers}/{total_prompts_evaluated})")
            
            if log_metric_name not in self._metrics["train"]:
                self._metrics["train"][log_metric_name] = []
            
            self._metrics["train"][log_metric_name].append(success_rate)
            self.log({log_metric_name: success_rate})

        return result

def integer_reward_func(prompts, completions, completion_ids, ground_truth, **kwargs):
    """
    Calculate rewards based on whether the answer is an integer.
    
    Args:
        prompts (List[str]): Original math questions
        completions (List[str]): Model completions
        completion_ids (List[List[int]]): Token IDs of completions
        ground_truth (List[int]): Correct answers
        
    Returns:
        List[float]: Rewards for each completion
    """
    rewards = []
    
    for completion in completions:
        reward = 0
        
        # Check if <answer></answer> tags exist
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, completion)
        
        if answer_match:
            # Extract the answer text
            answer_text = answer_match.group(1).strip()
            
            # Check if the answer is an integer
            try:
                # Convert potential float strings to int if they represent whole numbers
                parsed_answer = float(answer_text.replace(',', ''))
                if parsed_answer.is_integer():
                    reward += 1  # for answer being an integer
            except (ValueError, TypeError):
                # Not an integer/float, no reward
                pass
        
        rewards.append(reward)
    
    return rewards

def correctness_and_strategy_reward_func(prompts, completions, completion_ids, ground_truth, **kwargs):
    """
    Calculate rewards based on the correctness of the answer and brevity of response.
    
    Args:
        prompts (List[str]): Original math questions
        completions (List[str]): Model completions
        completion_ids (List[List[int]]): Token IDs of completions
        ground_truth (List[int]): Correct answers
        
    Returns:
        List[float]: Rewards for each completion
    """
    rewards = []
    
    for completion, completion_id, gt in zip(completions, completion_ids, ground_truth):
        reward = 0
        
        # Check if <answer></answer> tags exist
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, completion)
        
        if answer_match:
            # Extract the answer text
            answer_text = answer_match.group(1).strip()
            
            try:
                # Convert potential float strings to int if they represent whole numbers
                parsed_answer = float(answer_text.replace(',', ''))
                if parsed_answer.is_integer():
                    parsed_answer = int(parsed_answer)
                    
                    # Check if the answer matches ground truth
                    if parsed_answer == gt:
                        reward += 1  # for correct answer
                        # Add bonus for shorter completions
                        reward += 5/len(completion_id)  # Bonus for brevity when correct
            except (ValueError, TypeError):
                # Not an integer/float, no reward
                pass
        
        rewards.append(reward)
    
    return rewards

def react_reward_func(prompts, completions, completion_ids, ground_truth, **kwargs):
    """
    Calculate rewards based on the ReAct pattern format.
    
    Args:
        prompts (List[str]): Original math questions
        completions (List[str]): Model completions
        completion_ids (List[List[int]]): Token IDs of completions
        ground_truth (List[int]): Correct answers
        
    Returns:
        List[float]: Rewards for each completion
    """
    rewards = []
    
    for completion in completions:
        reward = 0
        
        # Check if the response starts with <think> tag
        if not completion.strip().startswith('<think>'):
            rewards.append(reward)
            continue
            
        # Check if the response ends with </answer> tag
        if not completion.strip().endswith('</answer>'):
            rewards.append(reward)
            continue
            
        # Check for adjacent </think><code> tags
        if '</think><code>' not in completion:
            rewards.append(reward)
            continue
            
        # Check if the tag before <answer> is </output> or </think>
        if not re.search(r'</output><answer>|</think><answer>', completion):
            rewards.append(reward)
            continue
        
        # If all conditions are met, give +1 reward
        reward += 1
        rewards.append(reward)
    
    return rewards


def main():
    # Initialize wandb (preferably on the main process in distributed training)
    # The GRPOTrainer will also handle wandb if `report_to="wandb"` is in args.
    # Explicit init here ensures project name is set as desired.
    # You might want to guard this with `if accelerator.is_main_process:` if using accelerate explicitly
    if os.environ.get("RANK", "0") == "0": # Basic check for main process
        wandb.init(project="ReAct_On_Roids")

    # Load and prepare datasets
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    eval_dataset = load_dataset("openai/gsm8k", "main", split="test")

    dataset_with_ground_truth = dataset.map(
        extract_ground_truth,
        remove_columns=["question", "answer"]
    )
    eval_dataset_with_ground_truth = eval_dataset.map(
        extract_ground_truth,
        remove_columns=["question", "answer"]
    )
    gsm300 = eval_dataset_with_ground_truth.shuffle(seed=42).select(range(300))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("August4293/Qwen2.5-0.5B-Instruct-with-output-tokens")

    # Prepare datasets for the agent
    prepared_dataset = prepare_data_for_e2b_agent(dataset_with_ground_truth, tokenizer, system_prompt=system_prompt, prompts_column="prompt")
    prepared_eval_dataset = prepare_data_for_e2b_agent(gsm300, tokenizer, system_prompt=system_prompt, prompts_column="prompt")

    # Initialize code executer and environment
    code_executer = E2BExecuter(api_key=os.environ["E2B_API_KEY"])
    
    my_env = CodeAgentEnvironment(
        code_executer=code_executer,
        tokenizer=tokenizer,
        parsing_string="<code>",
        stop_string="</code>",
    )

    # Evaluation generation configuration
    eval_generation_config = VLLMClientGenerationConfig(
        n=1,
        temperature=0.9,
        max_tokens=1024,
        repetition_penalty=1.2,
        top_p=0.9,
        top_k=50,
        min_p=0,
    )

    # Training arguments
    training_args = GRPOConfig(
        output_dir="Qwen_0.5B-GSM8K-Agent",
        reward_weights=[1,1,0.3],
        use_vllm=True,
        report_to="wandb",
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=1,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        num_iterations=3,
        logging_steps=1,
        log_completions=True,
        bf16=True,
        save_steps=40,
        save_total_limit=1,
        use_liger_loss=True,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = GRPOTrainerWithEval(
        model="August4293/Qwen2.5-0.5B-Instruct-with-output-tokens",
        reward_funcs=[react_reward_func,correctness_and_strategy_reward_func,integer_reward_func],
        args=training_args,
        train_dataset=prepared_dataset,
        environment=my_env,
        eval_dataset=prepared_eval_dataset,
        eval_gen_config=eval_generation_config,
        eval_every_n_steps=40,
    )

    # Start training
    trainer.train()

    # Upload to Hub
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
