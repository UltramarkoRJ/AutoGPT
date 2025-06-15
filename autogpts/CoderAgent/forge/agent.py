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


class CoderForgeAgent(Agent):
    """
    The Coder Agent is responsible for generating code based on task specifications
    provided by the Orchestrator or Planner.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        # Setup for CoderAgent, if any specific initialization is needed
        self.prompt_engine = PromptEngine("gpt-3.5-turbo") # Example, or a model better suited for coding

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # This method might not be directly used if Orchestrator calls specific methods.
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 CoderAgent task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    def _create_code_generation_prompt(self, task_specification: dict, user_goal: str) -> str:
        """
        Generates a prompt for the LLM to write code based on the task spec.
        """
        # Extract details from task_specification
        task_name = task_specification.get("name", "N/A")
        task_description = task_specification.get("description", "N/A")
        specifications = task_specification.get("specifications", "N/A")
        # expected_output_filename = task_specification.get("expected_output_filename", "script.py") # Could be used to guide language or structure

        prompt = f"""
        You are an expert software developer. Your role is to write code to fulfill the given task.
        The overall user goal for this project is: "{user_goal}"

        Current Coding Task: "{task_name}"
        Detailed Description: "{task_description}"
        Specific Requirements/Specifications: "{specifications}"

        Based on these details, please generate the necessary code.
        - If a specific programming language is implied by the specifications or filename (e.g. .py for Python, .js for JavaScript), use that language. Default to Python if unsure.
        - Ensure the code is functional and adheres to best practices.
        - Only output the raw code. Do not include any explanations, markdown formatting (like ```python), or additional text around the code.
        - If the task is to create a function, provide only the function definition. If it's a script, provide the full script.

        Generate the code now:
        """
        return prompt

    async def generate_code_for_task(self, task_specification: dict, user_goal: str) -> str:
        """
        Uses an LLM to generate code for the given task_specification.
        """
        if not task_specification:
            LOG.warning("Task specification is empty. Cannot generate code.")
            return "# Error: No task specification provided."

        code_prompt = self._create_code_generation_prompt(task_specification, user_goal)

        try:
            chat_completion_kwargs = {
                "messages": [
                    {"role": "system", "content": "You are an expert code generation assistant."},
                    {"role": "user", "content": code_prompt},
                ],
                "model": "gpt-4-turbo-preview", # Using a more capable model for code generation
                # "temperature": 0.2, # Lower temperature for more deterministic code
            }

            LOG.info(f"CoderAgent: Attempting LLM call for task: {task_specification.get('name', 'N/A')}")
            response = await chat_completion_request(**chat_completion_kwargs)

            generated_code = response["choices"][0]["message"]["content"]
            # LLMs might still wrap code in markdown, try to strip it
            if generated_code.startswith("```") and generated_code.endswith("```"):
                lines = generated_code.split('\n')
                if len(lines) > 1: # Check if there's a language hint like ```python
                    generated_code = '\n'.join(lines[1:-1]) # Remove first and last lines
                else: # Single line of ```code```
                    generated_code = generated_code.strip("`")


            LOG.info(f"CoderAgent: Successfully generated code for task: {task_specification.get('name', 'N/A')}")
            # LOG.debug(f"Generated code: \n{generated_code}") # Be careful logging full code
            return generated_code

        except Exception as e:
            LOG.error(f"CoderAgent: An unexpected error occurred during code generation: {e}")
            return f"# Error: An unexpected error occurred during code generation: {e}"

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        # This execute_step is for standalone CoderAgent operation.
        # Orchestrator will likely call generate_code_for_task directly.
        LOG.info(f"CoderAgent executing step for task_id: {task_id}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        # Assuming step_request.input is a JSON string containing 'task_specification' and 'user_goal'
        try:
            input_data = json.loads(step_request.input)
            task_spec = input_data.get("task_specification")
            user_goal = input_data.get("user_goal", "Not specified in this step's direct input.")
        except json.JSONDecodeError:
            LOG.error("CoderAgent: Invalid JSON input for execute_step.")
            task_spec = None
            user_goal = "Error: Invalid input."

        if not task_spec:
            LOG.error("CoderAgent: No task_specification provided in step_request.")
            step.output = "Error: No task_specification provided to CoderAgent."
            step.is_last = True
            return step

        generated_code = await self.generate_code_for_task(task_spec, user_goal)

        # Save the generated code as an artifact
        # Determine filename:
        task_name_for_file = task_spec.get("name", "untitled_code").replace(" ", "_").lower()
        extension = ".py" # Default, could be inferred from task_spec or LLM output analysis
        if "python" in task_spec.get("description","").lower() or "python" in task_spec.get("specifications","").lower():
            extension = ".py"
        elif "javascript" in task_spec.get("description","").lower() or "javascript" in task_spec.get("specifications","").lower():
            extension = ".js"

        code_artifact_name = f"generated_code_{task_name_for_file}_{task_id}{extension}"

        try:
            self.workspace.write(task_id=task_id, path=code_artifact_name, data=generated_code.encode())
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=code_artifact_name,
                relative_path="",
                agent_created=True,
            )
            LOG.info(f"CoderAgent saved generated code to {code_artifact_name}")
            step.output = f"Code generated by CoderAgent for task '{task_spec.get('name', 'N/A')}'. Saved as {code_artifact_name}."
        except Exception as e:
            LOG.error(f"CoderAgent: Error saving code artifact: {e}")
            step.output = f"Error saving generated code: {e}"

        step.is_last = True # CoderAgent's job for this step is done.
        return step
