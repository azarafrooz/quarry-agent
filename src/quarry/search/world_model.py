"""
The world model for the Quarry:
Automated insight discovery via Python code generation, using f-string formatted SQL and composable code.
"""

from .base import WorldModel, SearchConfig
import copy
from ..types import ReActState
from ..agent import AbstractPythonCodeAgent

from ..resources.prompts import (
    REACT_PROMPT,
    format_reward_prompt,
    extract_score,
    append_kernel_output,
    is_action_complete,
    extract_python_codes,
    trim_after_first_python_code_block,
)


class ReActWorldModel(WorldModel):
    """Quarry World Model
    State: (step_idx,  action, goal)
    action : "Contains the LLM response composed of thoughts and codes
    """

    def __init__(
        self,
        agent: AbstractPythonCodeAgent,
        goal: str,
        max_iterations: int = 6,
        batch_size=2,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.agent = agent
        self.batch_size = batch_size
        self.goal = goal

    def init_state(self) -> ReActState:
        """Initialize the world model with a React Prompt and initial state
        :return: the initial state
        """
        action = ""
        values = {
            "<<dialect>>": self.agent.dialect,
            "<<max_iterations>>": str(self.max_iterations),
            "<<tool_descriptions>>": self.agent.get_relevant_tools(self.goal),
            # "<<context>>": f"\n{'-'.join(self.agent.get_related_documentation(self.goal))}",
            "<<context>>": f"{self.agent.get_schema()}"
            + f"\n{'-'.join(self.agent.get_related_documentation(self.goal))}",
            "<<goal>>": self.goal,
        }
        prompt = REACT_PROMPT
        for k, v in values.items():
            prompt = prompt.replace(k, v)

        return ReActState(
            step_idx=0, prompt=prompt, action=action, goal=self.goal, finished=False
        )

    def step(self, state: ReActState, action: str) -> tuple[ReActState, dict]:
        """Take a step in the world model.
        :param state: the current state
        :param action: the action to take.
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        step_idx = state.step_idx
        prompt = state.prompt
        finished = is_action_complete(action)
        state = ReActState(
            step_idx=step_idx + 1,
            prompt=prompt,
            action=state.action + "\n" + action,
            goal=self.goal,
            finished=finished,
        )

        return (
            state,
            {"goal_reached": finished},
        )  # TODO: For now assume we never reach an explicit goal unless a specific question is given. In that case, we need extra LLM or something

    def is_terminal(self, state: ReActState) -> bool:
        if state.step_idx == self.max_iterations - 1 or state.finished:
            return True
        return False


class ReActConfig(SearchConfig):
    def __init__(
        self,
        agent: AbstractPythonCodeAgent,
        num_actions: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        batch_size: int = 1,
        reward_alpha: float = 0.5,
        goal_reward_default: float = 0.0,
        goal_reached_reward: float = 100.0,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.num_actions = num_actions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def get_actions(self, state: ReActState) -> list[str]:
        """
        Actions are the text generated during the react
        """
        prompt = state.prompt
        if not state.action:
            messages = [
                self.agent.user_message(prompt),
            ]
        else:
            messages = [
                self.agent.user_message(prompt),
                self.agent.assistant_message(state.action),
            ]

        actions = []
        for _ in range(
            2 * self.num_actions if state.step_idx == 0 else self.num_actions
        ):
            llm_response = self.agent.submit_prompt(
                messages,
                # max_new_tokens=self.max_new_tokens,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                # do_sample=True,
            )
            action = trim_after_first_python_code_block(llm_response)
            codes = extract_python_codes(action)
            if codes:
                code = codes[0]  # first code is all we care
                past_codes = extract_python_codes(state.action)  # the codes so far
                with self.agent.preserve_state():
                    # reload the previous state of python interpreter
                    for past_code in past_codes:
                        self.agent.execute_code(past_code)
                    kernel_output = self.agent.execute_code(code)
                action = append_kernel_output(action, kernel_output)
            actions.append(action)

        return actions

    def fast_reward(self, state: ReActState, action: str) -> tuple[float, dict]:
        analysis = state.prompt + "\n" + action
        intuition = 0  # could be a logit or mutual information.
        successful_run_score = 0  # similar to intuition , we encourage successful runs.
        reward_prompt = format_reward_prompt(analysis)
        eval_response = self.agent.submit_prompt(
            [
                self.agent.user_message(reward_prompt),
            ],
            max_tokens=512,
        )
        self_eval_score = extract_score(eval_response)
        self_eval_score = (1 + self_eval_score) / 2
        print(f"eval score is: {self_eval_score}")

        return (
            self.calculate_reward(intuition + successful_run_score, self_eval_score),
            {
                "intuition": intuition + successful_run_score,
                "self_eval": self_eval_score,
            },
        )

    def calculate_reward(self, intuition, self_eval, goal_reached=None) -> float:
        goal_reward = self.goal_reward_default
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (
            1 - self.reward_alpha
        )

    def reward(
        self,
        state: ReActState,
        action: str,
        intuition: float = None,
        self_eval: float = None,
        goal_reached: tuple[bool, float] = None,
    ) -> tuple[float, dict]:
        return (
            self.calculate_reward(intuition, self_eval, goal_reached),
            {"intuition": intuition, "goal_reached": goal_reached},
        )
