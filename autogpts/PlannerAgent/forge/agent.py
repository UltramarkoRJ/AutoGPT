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


class ForgeAgent(Agent):
    """
    The Planner Agent is responsible for taking a high-level user goal
    and breaking it down into a sequence of actionable tasks for other
    specialized agents.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        # Setup for PlannerAgent, if any specific initialization is needed
        # For example, initializing a specific prompt engine or memory for planning
        self.prompt_engine = PromptEngine("gpt-3.5-turbo") # Example, choose as needed

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 PlannerAgent task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    def _create_initial_plan_prompt(self, user_goal: str) -> str:
        """
        Generates a prompt for the LLM to create a plan from the user goal.
        """
        prompt = f"""
        You are an expert planning assistant. Your role is to break down a given user goal into a sequence of tasks that can be executed by a team of specialized AI agents.

        User Goal: "{user_goal}"

        Based on this goal, create a detailed plan as a JSON array of objects. Each object in the array represents a task and must have the following fields:
        - "task_id": A unique string identifier for the task (e.g., "task-1", "task-2").
        - "name": A short, descriptive name for the task (e.g., "Gather Requirements", "Develop Feature X").
        - "description": A detailed explanation of what the task entails, providing enough context for another agent to execute it.
        - "agent_role": The suggested specialized agent role to handle this task (e.g., "ResearcherAgent", "CoderAgent", "ReviewerAgent", "QAEngineerAgent", "DocumentationWriterAgent").
        - "dependencies": A list of "task_id" strings that this task depends on. An empty list means no dependencies.
        - "status": Initialize this to "pending".
        - "artifacts": An empty list, which will be populated later with filenames of outputs from this task.
        - "specifications": A placeholder string or object for detailed task specifications. For now, you can put "To be defined by Planner or subsequent tasks."

        Ensure the tasks are logically sequenced. The first task(s) should typically have no dependencies.
        Return ONLY the JSON array. Do not include any other text, explanations, or markdown formatting around the JSON.

        Example of a single task object:
        {{
            "task_id": "task-1",
            "name": "Example Task",
            "description": "This is an example task description.",
            "agent_role": "ResearcherAgent",
            "dependencies": [],
            "status": "pending",
            "artifacts": [],
            "specifications": "Research topic X based on requirements Y."
        }}

        Now, generate the plan for the user goal: "{user_goal}"
        """
        return prompt

    async def generate_plan_from_goal(self, user_goal: str, task_id: str, step_id: str) -> list:
        """
        Uses an LLM to generate a plan from the user_goal.
        """
        if not user_goal:
            LOG.warning("User goal is empty. Cannot generate plan.")
            return []

        plan_prompt = self._create_initial_plan_prompt(user_goal)

        try:
            chat_completion_kwargs = {
                "messages": [
                    {"role": "system", "content": "You are an expert planning assistant that outputs JSON."},
                    {"role": "user", "content": plan_prompt},
                ],
                "model": "gpt-3.5-turbo", # Ensure this model is available or use a configured one
                # "temperature": 0.7, # Adjust as needed for creativity vs determinism
            }

            LOG.info(f"Attempting LLM call with prompt for goal: {user_goal[:100]}")
            # Assuming a method like this exists based on Forge's capabilities
            # This might need adjustment based on the actual LLM interaction methods provided by the Forge SDK
            # For now, we'll use the generic chat_completion_request which is part of the SDK imports
            response = await chat_completion_request(**chat_completion_kwargs)

            llm_output = response["choices"][0]["message"]["content"]
            LOG.info(f"LLM raw output: {llm_output}")

            # Attempt to parse the JSON output
            # The prompt asks for JSON only, but LLMs can sometimes add markdown
            if llm_output.startswith("```json"):
                llm_output = llm_output.strip("```json").strip("`").strip()

            plan = json.loads(llm_output)
            LOG.info(f"Successfully parsed plan from LLM: {pprint.pformat(plan)}")
            return plan

        except json.JSONDecodeError as e:
            LOG.error(f"Failed to decode JSON from LLM output: {e}")
            LOG.error(f"LLM output that caused error: {llm_output}")
            # Fallback or error handling: Maybe return a single task to report the error
            error_task = {
                "task_id": "error-plan-generation", "name": "Plan Generation Failed",
                "description": f"LLM failed to generate a valid JSON plan. Error: {e}. Raw output: {llm_output}",
                "agent_role": "PlannerAgent", "dependencies": [], "status": "failed", "artifacts": [], "specifications": ""
            }
            return [error_task]
        except Exception as e:
            LOG.error(f"An unexpected error occurred during plan generation: {e}")
            LOG.error(f"Details: Task ID {task_id}, Step ID {step_id}")
            # Fallback or error handling
            error_task = {
                "task_id": "error-unexpected", "name": "Unexpected Planning Error",
                "description": f"An unexpected error occurred. Error: {e}.",
                "agent_role": "PlannerAgent", "dependencies": [], "status": "failed", "artifacts": [], "specifications": ""
            }
            return [error_task]


    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        LOG.info(f"PlannerAgent executing step for task_id: {task_id}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        user_goal = step_request.input
        if not user_goal: # If input is not directly provided in step_request, get from task
            task = await self.db.get_task(task_id)
            user_goal = task.input

        LOG.info(f"PlannerAgent received user goal: {user_goal}")

        if not user_goal:
            LOG.error("No user goal provided to PlannerAgent.")
            step.output = "Error: No user goal provided to generate a plan."
            step.is_last = True
            return step

        # Generate the plan using the LLM
        generated_plan = await self.generate_plan_from_goal(user_goal, task_id, step.step_id)

        # Save the generated plan as an artifact
        plan_artifact_name = f"generated_plan_for_goal_{task_id}.json"
        try:
            plan_json_data = json.dumps(generated_plan, indent=4)
            self.workspace.write(task_id=task_id, path=plan_artifact_name, data=plan_json_data.encode())
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=plan_artifact_name,
                relative_path="",
                agent_created=True,
            )
            LOG.info(f"PlannerAgent saved generated plan to {plan_artifact_name}")
            step.output = f"Plan generated by PlannerAgent for goal '{user_goal}'. Saved as {plan_artifact_name}."
            # This output (the plan itself) might be useful for the Orchestrator directly
            # For Agent Protocol, the primary output is a string, but artifacts handle complex data.
            # We could also put the plan_json_data into step.additional_output if that's supported/useful.
        except Exception as e:
            LOG.error(f"Error saving plan artifact: {e}")
            step.output = f"Error saving generated plan: {e}"

        step.is_last = True # PlannerAgent's job for this step is done once plan is generated.
        return step
