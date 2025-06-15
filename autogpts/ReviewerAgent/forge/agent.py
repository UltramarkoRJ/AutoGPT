from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
    PromptEngine,
    chat_completion_request,
    ChromaMemStore
)
import json
import pprint

LOG = ForgeLogger(__name__)


class ReviewerForgeAgent(Agent):
    """
    The Reviewer Agent is responsible for reviewing artifacts produced by other agents,
    such as code from the CoderAgent, based on task specifications.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        self.prompt_engine = PromptEngine("gpt-3.5-turbo") # Or a model suitable for review/analysis

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 ReviewerAgent task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    def _create_code_review_prompt(self, code_to_review: str, task_specification: dict, user_goal: str) -> str:
        """
        Generates a prompt for the LLM to review code.
        """
        task_name = task_specification.get("name", "N/A")
        task_description = task_specification.get("description", "N/A")
        specifications = task_specification.get("specifications", "N/A")

        prompt = f"""
        You are an expert code reviewer. Your task is to review the provided code based on the original task specification and the overall user goal.

        Overall User Goal: "{user_goal}"
        Original Task for the Code: "{task_name}"
        Original Task Description: "{task_description}"
        Original Task Specifications: "{specifications}"

        Code to Review:
        ```
        {code_to_review}
        ```

        Please provide your review, including:
        1. A brief summary of what the code does.
        2. An assessment of whether the code seems to address the main points of the original task specification.
        3. Identification of any obvious potential issues, bugs, or areas for improvement (e.g., clarity, efficiency, security). You don't need to execute the code, rely on static analysis and your expertise.
        4. Suggestions for improvement if any.
        5. A simple overall assessment (e.g., "Looks good", "Needs minor revisions", "Needs major revisions").

        Present the review as well-structured text.
        Generate the code review now:
        """
        return prompt

    async def review_code(self, code_to_review: str, task_specification: dict, user_goal: str) -> str:
        """
        Uses an LLM to review the provided code.
        """
        if not code_to_review:
            LOG.warning("No code provided for review.")
            return "Error: No code provided for review."
        if not task_specification:
            LOG.warning("No task specification provided for context. Review will be limited.")
            # Allow review to proceed but with a warning, or return an error
            # return "Error: No task specification provided for review context."

        review_prompt = self._create_code_review_prompt(code_to_review, task_specification, user_goal)

        try:
            chat_completion_kwargs = {
                "messages": [
                    {"role": "system", "content": "You are an expert code review assistant."},
                    {"role": "user", "content": review_prompt},
                ],
                "model": "gpt-4-turbo-preview", # A capable model for code understanding
            }

            LOG.info(f"ReviewerAgent: Attempting LLM call for code review of task: {task_specification.get('name', 'N/A')}")
            response = await chat_completion_request(**chat_completion_kwargs)

            review_text = response["choices"][0]["message"]["content"]
            LOG.info(f"ReviewerAgent: Successfully generated review for task: {task_specification.get('name', 'N/A')}")
            return review_text

        except Exception as e:
            LOG.error(f"ReviewerAgent: An unexpected error occurred during code review: {e}")
            return f"Error: An unexpected error occurred during code review: {e}"

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        LOG.info(f"ReviewerAgent executing step for task_id: {task_id}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        try:
            # Assuming step_request.input is a JSON string with 'code_to_review', 'task_specification', and 'user_goal'
            input_data = json.loads(step_request.input)
            code_to_review = input_data.get("code_to_review")
            task_spec = input_data.get("task_specification")
            user_goal = input_data.get("user_goal", "Not specified in step input.")
        except json.JSONDecodeError:
            LOG.error("ReviewerAgent: Invalid JSON input for execute_step.")
            step.output = "Error: Invalid JSON input for ReviewerAgent."
            step.is_last = True
            return step

        if not code_to_review or not task_spec:
            LOG.error("ReviewerAgent: Missing 'code_to_review' or 'task_specification' in input.")
            step.output = "Error: Missing 'code_to_review' or 'task_specification'."
            step.is_last = True
            return step

        review_output = await self.review_code(code_to_review, task_spec, user_goal)

        artifact_name = f"code_review_{task_spec.get('name', 'unknown_task').replace(' ','_')}_{task_id}.txt"
        try:
            self.workspace.write(task_id=task_id, path=artifact_name, data=review_output.encode())
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=artifact_name,
                relative_path="",
                agent_created=True,
            )
            LOG.info(f"ReviewerAgent saved code review to {artifact_name}")
            step.output = f"Code review completed by ReviewerAgent for task '{task_spec.get('name', 'N/A')}'. Saved as {artifact_name}."
        except Exception as e:
            LOG.error(f"ReviewerAgent: Error saving review artifact: {e}")
            step.output = f"Error saving code review: {e}"

        step.is_last = True
        return step
