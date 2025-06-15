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


class ResearcherForgeAgent(Agent):
    """
    The Researcher Agent is responsible for gathering and synthesizing information
    on a given topic, guided by an overall user goal.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        self.prompt_engine = PromptEngine("gpt-3.5-turbo") # Or a model good for summarization/research

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # This method might not be directly used if Orchestrator calls specific methods.
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 ResearcherAgent task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    def _create_research_prompt(self, research_topic: str, user_goal: str) -> str:
        """
        Generates a prompt for the LLM to perform research on the topic.
        """
        prompt = f"""
        You are an expert researcher. Your task is to gather and synthesize information on the given research topic, keeping in mind the overall user goal.

        Overall User Goal: "{user_goal}"
        Research Topic: "{research_topic}"

        Please provide a concise summary of your findings. This could include:
        - Key facts and figures.
        - Definitions of important terms.
        - Potential approaches or solutions related to the topic if applicable.
        - If the topic implies looking for tools or technologies, list some examples and their primary uses.

        Focus on information directly relevant to the research topic.
        Present the findings as a well-structured text.
        You do not have live internet access, so base your findings on your existing knowledge up to your last training cut-off. Act as if you have performed a comprehensive search and are summarizing the key information.

        Generate the research summary now:
        """
        return prompt

    async def perform_research(self, research_topic: str, user_goal: str) -> str:
        """
        Uses an LLM to gather and synthesize information on the research_topic.
        """
        if not research_topic:
            LOG.warning("Research topic is empty. Cannot perform research.")
            return "Error: No research topic provided."

        research_prompt = self._create_research_prompt(research_topic, user_goal)

        try:
            chat_completion_kwargs = {
                "messages": [
                    {"role": "system", "content": "You are an expert research assistant."},
                    {"role": "user", "content": research_prompt},
                ],
                "model": "gpt-3.5-turbo-16k", # Using a model with a larger context window if research is extensive
            }

            LOG.info(f"ResearcherAgent: Attempting LLM call for topic: {research_topic[:100]}")
            response = await chat_completion_request(**chat_completion_kwargs)

            research_findings = response["choices"][0]["message"]["content"]
            LOG.info(f"ResearcherAgent: Successfully gathered research for topic: {research_topic[:100]}")
            return research_findings

        except Exception as e:
            LOG.error(f"ResearcherAgent: An unexpected error occurred during research: {e}")
            return f"Error: An unexpected error occurred during research: {e}"

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        # This execute_step is for standalone ResearcherAgent operation.
        LOG.info(f"ResearcherAgent executing step for task_id: {task_id}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        try:
            # Assuming step_request.input is a JSON string with 'research_topic' and 'user_goal'
            input_data = json.loads(step_request.input)
            research_topic = input_data.get("research_topic")
            user_goal = input_data.get("user_goal", "Not specified in step input.")
        except json.JSONDecodeError:
            LOG.error("ResearcherAgent: Invalid JSON input for execute_step.")
            research_topic = None
            user_goal = "Error: Invalid input."

        if not research_topic: # Fallback if not in JSON, try direct input.
             task = await self.db.get_task(task_id)
             # This assumes the direct input to the task IS the research topic.
             # A more robust way would be a structured input for the task itself.
             research_topic = task.input
             # user_goal might need to be passed differently if not in step_request.input
             # For now, we'll rely on it being part of the JSON or a default.

        if not research_topic:
            LOG.error("ResearcherAgent: No research_topic provided.")
            step.output = "Error: No research_topic provided to ResearcherAgent."
            step.is_last = True
            return step

        research_findings = await self.perform_research(research_topic, user_goal)

        artifact_name = f"research_summary_{task_id}.txt"
        try:
            self.workspace.write(task_id=task_id, path=artifact_name, data=research_findings.encode())
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=artifact_name,
                relative_path="",
                agent_created=True,
            )
            LOG.info(f"ResearcherAgent saved research findings to {artifact_name}")
            step.output = f"Research findings gathered by ResearcherAgent for topic '{research_topic}'. Saved as {artifact_name}."
        except Exception as e:
            LOG.error(f"ResearcherAgent: Error saving research artifact: {e}")
            step.output = f"Error saving research findings: {e}"

        step.is_last = True
        return step
