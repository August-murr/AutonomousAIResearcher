# Autonomous AI Researcher

**⚠️ Experimental Project - Early Stage & Seeking Collaboration ⚠️**
## Goal

This is an early-stage, highly ambitious research project exploring the potential of training autonomous AI agents using Reinforcement Learning (RL) to perform increasingly complex tasks in Software Engineering (SWE), Machine Learning (ML) engineering, and eventually AI research. The goal is to develop and open-source an autonomous agent that is capable of conducting AI research and development tasks, with the ultimate aim of accelerating AI progress through self-improvement.

---


## Our Approach

This project is built on the idea that an AI researcher agent can be trained iteratively using RL:

1.  **Autonomous Agents:** Utilizing foundation models fine-tuned with RL algorithms (starting with methods like GRPO, exploring others) to act within defined environments.
2.  **Evolving Environments:** Starting with sandboxed code execution environments (with GPU access) and progressively adding tools like web search, API access, database interaction, and eventually, full control over code repositories.
3.  **Progressive Task Datasets:** Curating a dataset of tasks that increase in complexity:
    * *Initial:* Simple coding tasks (e.g., LeetCode-style with GPU usage).
    * *Intermediate:* Standard ML engineering tasks (data preprocessing, model fine-tuning, pipeline optimization).
    * *Advanced:* Research-oriented tasks (reimplementing papers, reproducing results, hypothesis generation, reporting findings).
4.  **Scalable Evaluation:** Moving beyond expensive, human-written test cases towards scalable methods like self-evaluation, self-verification, and self-ranking to provide reward signals for the RL agent.
5.  **Focused Training:** Employing RL strategies that prioritize tasks on the agent's learning "horizon" – challenging but achievable tasks where reward signals can be obtained, potentially using techniques like self-ranking to handle sparse rewards effectively.

## Detailed Plan & Progress Tracking

For a more detailed breakdown of tasks, ongoing work, and progress tracking, please visit the **[GitHub Project Board](https://github.com/users/August-murr/projects/2)**. 

---

## How to Contribute
1.  **Check the [Issues Tab](https://github.com/August-murr/AutonomousAIResearcher/issues)**: This is the primary place to find specific tasks, bug reports, and feature requests that need work. 
2.  **Start a Conversation:** Have an idea? Feedback? A question? Don't hesitate to:
    * Open a **New Issue** for specific proposals, bugs, or feature requests.
    * Use the **[Discussions Tab](https://github.com/August-murr/AutonomousAIResearcher/discussions)**
