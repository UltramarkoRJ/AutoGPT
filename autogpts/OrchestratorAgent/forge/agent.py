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
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    # --- Placeholder functions for specialized agents ---
    async def _delegate_to_planner(self, task_id: str, step_id: str, current_task: dict):
        LOG.info(f"Delegating task '{current_task['task_id']}' to PlannerAgent: {current_task['description']}")
        delegated_task_artifact_name = f"delegated_task_planner_{current_task['task_id']}.json"
        self.workspace.write(task_id=task_id, path=delegated_task_artifact_name, data=json.dumps(current_task, indent=4).encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=delegated_task_artifact_name, relative_path="", agent_created=True)

        # Simulate planner output
        output_artifact_name = f"output_planner_{current_task['task_id']}.txt"
        planner_output = f"Detailed plan for '{current_task['description']}' created by PlannerAgent."
        self.workspace.write(task_id=task_id, path=output_artifact_name, data=planner_output.encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_name, relative_path="", agent_created=True)
        LOG.info(f"PlannerAgent completed task '{current_task['task_id']}'. Output: {output_artifact_name}")
        return {"output_artifact": output_artifact_name, "details": planner_output}

    async def _delegate_to_researcher(self, task_id: str, step_id: str, current_task: dict, user_goal: str):
        LOG.info(f"Orchestrator: Delegating task '{current_task['task_id']}' to ResearcherAgent: {current_task['description']}")

        delegated_task_spec_artifact_name = f"delegated_spec_researcher_{current_task['task_id']}.json"
        self.workspace.write(task_id=task_id, path=delegated_task_spec_artifact_name, data=json.dumps(current_task, indent=4).encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=delegated_task_spec_artifact_name, relative_path="", agent_created=True)

        research_findings_str = ""
        output_artifact_filename = f"research_summary_{current_task['task_id']}.txt"

        try:
            from autogpts.ResearcherAgent.forge.agent import ResearcherForgeAgent
            researcher_agent = ResearcherForgeAgent(self.db, self.workspace) # Researcher might also benefit from its own workspace

            LOG.info(f"Orchestrator: Instantiated ResearcherAgent for task {current_task['task_id']}.")

            research_topic = current_task.get("description") # Default to using the task description as the topic
            if isinstance(current_task.get("specifications"), dict) and current_task["specifications"].get("topic"):
                research_topic = current_task["specifications"]["topic"]
            elif isinstance(current_task.get("specifications"), str) and current_task["specifications"]: # If spec is just a string
                research_topic = current_task["specifications"]


            if not research_topic:
                LOG.error(f"Orchestrator: No research topic found for task {current_task['task_id']}.")
                research_findings_str = "Error: No research topic specified for the task."
            else:
                research_findings_str = await researcher_agent.perform_research(
                    research_topic=research_topic,
                    user_goal=user_goal
                )

            if research_findings_str.startswith("Error:"):
                LOG.error(f"ResearcherAgent returned an error for task {current_task['task_id']}: {research_findings_str}")
            else:
                LOG.info(f"Orchestrator: ResearcherAgent successfully gathered information for task {current_task['task_id']}.")

            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=research_findings_str.encode())
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step_id,
                file_name=output_artifact_filename,
                relative_path="",
                agent_created=True
            )
            return {"output_artifact": output_artifact_filename, "details": f"Research for {current_task['task_id']} attempted."}

        except ImportError as ie:
            LOG.error(f"Orchestrator: Failed to import ResearcherAgent for task {current_task['task_id']}: {ie}.")
            error_content = f"Error: ResearcherAgent could not be imported: {ie}"
            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
            await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
            return {"output_artifact": output_artifact_filename, "details": "ResearcherAgent import failed."}
        except Exception as e:
            LOG.error(f"Orchestrator: An error occurred while delegating to ResearcherAgent for task {current_task['task_id']}: {e}")
            error_content = f"Error: Exception during ResearcherAgent delegation: {e}"
            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
            await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
            return {"output_artifact": output_artifact_filename, "details": f"Exception during ResearcherAgent delegation: {e}"}

    async def _delegate_to_coder(self, task_id: str, step_id: str, current_task: dict, user_goal: str):
        LOG.info(f"Orchestrator: Delegating task '{current_task['task_id']}' to CoderAgent: {current_task['description']}")

        # Create an artifact representing the task specification for the CoderAgent (optional, for clarity)
        delegated_task_spec_artifact_name = f"delegated_spec_coder_{current_task['task_id']}.json"
        self.workspace.write(task_id=task_id, path=delegated_task_spec_artifact_name, data=json.dumps(current_task, indent=4).encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=delegated_task_spec_artifact_name, relative_path="", agent_created=True)

        generated_code_str = ""
        # Attempt to determine a reasonable default filename, CoderAgent might refine this too.
        # For example, from a 'target_filename' field in current_task['specifications'] or current_task['artifacts']
        default_filename = f"generated_code_{current_task['task_id']}.py"
        if isinstance(current_task.get('specifications'), dict) and current_task['specifications'].get('target_filename'):
            default_filename = current_task['specifications']['target_filename']
        elif current_task.get('name'):
             safe_name = current_task['name'].replace(" ", "_").replace("/", "_").lower()
             default_filename = f"{safe_name}_{current_task['task_id']}.py"


        output_artifact_filename = default_filename


        try:
            from autogpts.CoderAgent.forge.agent import CoderForgeAgent
            coder_agent = CoderForgeAgent(self.db, self.workspace) # Coder might need its own workspace in a more advanced setup

            LOG.info(f"Orchestrator: Instantiated CoderAgent for task {current_task['task_id']}.")

            # Provide the whole current_task as task_specification, CoderAgent can extract what it needs
            generated_code_str = await coder_agent.generate_code_for_task(
                task_specification=current_task,
                user_goal=user_goal
            )

            if generated_code_str.startswith("# Error:"):
                LOG.error(f"CoderAgent returned an error for task {current_task['task_id']}: {generated_code_str}")
                # Save the error message as the artifact content
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=generated_code_str.encode())
            else:
                # If CoderAgent doesn't explicitly name its output, use the determined filename
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=generated_code_str.encode())
                LOG.info(f"Orchestrator: CoderAgent successfully generated code for task {current_task['task_id']}. Saved to {output_artifact_filename}")

            await self.db.create_artifact(
                task_id=task_id,
                step_id=step_id,
                file_name=output_artifact_filename, # This is the artifact name
                relative_path="", # Assuming it's in the root of the task's workspace artifacts
                agent_created=True # This artifact is created by Orchestrator from Coder's output
            )
            return {"output_artifact": output_artifact_filename, "details": f"Code generation for {current_task['task_id']} attempted."}

        except ImportError as ie:
            LOG.error(f"Orchestrator: Failed to import CoderAgent for task {current_task['task_id']}: {ie}. Coder task cannot be performed.")
            error_content = f"# Error: CoderAgent could not be imported: {ie}"
            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
            await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
            return {"output_artifact": output_artifact_filename, "details": "CoderAgent import failed."}
        except Exception as e:
            LOG.error(f"Orchestrator: An error occurred while delegating to CoderAgent for task {current_task['task_id']}: {e}")
            error_content = f"# Error: Exception during CoderAgent delegation: {e}"
            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
            await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
            return {"output_artifact": output_artifact_filename, "details": f"Exception during CoderAgent delegation: {e}"}

    async def _delegate_to_reviewer(self, task_id: str, step_id: str, current_task: dict, user_goal: str, plan: list):
        LOG.info(f"Orchestrator: Delegating task '{current_task['task_id']}' to ReviewerAgent: {current_task['description']}")

        output_artifact_filename = f"code_review_{current_task['task_id']}.txt"
        error_content = ""
        code_to_review_content = ""
        original_coder_task_spec = None

        try:
            # Identify the input code artifact from dependencies
            if not current_task.get("dependencies"):
                error_content = f"Error: Reviewer task '{current_task['task_id']}' has no dependencies to find code artifact from."
                LOG.error(error_content)
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
                await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
                return {"output_artifact": output_artifact_filename, "details": "Review task dependency error."}

            # Assuming the first dependency is the Coder task that produced the code
            coder_task_id = current_task["dependencies"][0]
            coder_task_from_plan = next((t for t in plan if t["task_id"] == coder_task_id), None)

            if not coder_task_from_plan:
                error_content = f"Error: Could not find dependent Coder task '{coder_task_id}' in the plan."
                LOG.error(error_content)
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
                await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
                return {"output_artifact": output_artifact_filename, "details": "Dependent Coder task not found."}

            code_artifact_filename = coder_task_from_plan.get("output")
            if not code_artifact_filename:
                error_content = f"Error: Coder task '{coder_task_id}' did not produce an output artifact for review."
                LOG.error(error_content)
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
                await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
                return {"output_artifact": output_artifact_filename, "details": "Coder task output artifact missing."}

            if not self.workspace.exists(task_id=task_id, path=code_artifact_filename):
                error_content = f"Error: Code artifact '{code_artifact_filename}' not found in workspace for task {task_id}."
                LOG.error(error_content)
                # It's possible the artifact was created for the original Coder task_id, not the Orchestrator's task_id
                # This needs careful handling of workspace contexts if agents have their own.
                # For now, assuming Orchestrator's workspace for all artifacts from delegated tasks.
                self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
                await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
                return {"output_artifact": output_artifact_filename, "details": "Code artifact not found."}

            code_to_review_content = self.workspace.read(task_id=task_id, path=code_artifact_filename).decode()
            original_coder_task_spec = coder_task_from_plan # This is the whole task dict for the Coder

            from autogpts.ReviewerAgent.forge.agent import ReviewerForgeAgent
            reviewer_agent = ReviewerForgeAgent(self.db, self.workspace)

            LOG.info(f"Orchestrator: Instantiated ReviewerAgent for task {current_task['task_id']}.")

            review_text = await reviewer_agent.review_code(
                code_to_review=code_to_review_content,
                task_specification=original_coder_task_spec, # Pass the Coder's task spec
                user_goal=user_goal
            )

            if review_text.startswith("Error:"):
                LOG.error(f"ReviewerAgent returned an error for task {current_task['task_id']}: {review_text}")
            else:
                LOG.info(f"Orchestrator: ReviewerAgent successfully reviewed code for task {current_task['task_id']}.")

            self.workspace.write(task_id=task_id, path=output_artifact_filename, data=review_text.encode())
            await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
            return {"output_artifact": output_artifact_filename, "details": f"Code review for {current_task['task_id']} attempted."}

        except ImportError as ie:
            LOG.error(f"Orchestrator: Failed to import ReviewerAgent: {ie}. Review task cannot be performed.")
            error_content = f"Error: ReviewerAgent could not be imported: {ie}"
        except FileNotFoundError as fnfe:
            LOG.error(f"Orchestrator: Code artifact not found for review: {fnfe}. This might indicate an issue with artifact path or Coder task output.")
            error_content = f"Error: Code artifact not found for review: {fnfe}."
        except Exception as e:
            LOG.error(f"Orchestrator: An error occurred while delegating to ReviewerAgent for task {current_task['task_id']}: {e}")
            error_content = f"Error: Exception during ReviewerAgent delegation: {e}"

        # Common error handling for exceptions caught above
        self.workspace.write(task_id=task_id, path=output_artifact_filename, data=error_content.encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_filename, relative_path="", agent_created=True)
        return {"output_artifact": output_artifact_filename, "details": f"Review task failed: {error_content}"}

    async def _finalize_task(self, task_id: str, step_id: str, current_task: dict):
        LOG.info(f"Finalizing task '{current_task['task_id']}': {current_task['description']}")
        # This is where the orchestrator would package the final solution.
        # For now, just log and create a dummy artifact.
        output_artifact_name = f"final_solution_{task_id}.txt"
        final_output = "All tasks completed. Solution packaged by OrchestratorAgent."
        self.workspace.write(task_id=task_id, path=output_artifact_name, data=final_output.encode())
        await self.db.create_artifact(task_id=task_id, step_id=step_id, file_name=output_artifact_name, relative_path="", agent_created=True)
        LOG.info(f"OrchestratorAgent finalized task '{current_task['task_id']}'. Output: {output_artifact_name}")
        return {"output_artifact": output_artifact_name, "details": final_output}

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        LOG.info(f"Executing step for task_id: {task_id} with input: {step_request.input}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        # Get the overall task goal
        task = await self.db.get_task(task_id)
        user_goal = task.input
        LOG.info(f"User goal for task {task_id}: {user_goal}")

        plan_artifact_name = "initial_plan.json"
        plan = []

        # 1. Load or Generate Plan
        try:
            if self.workspace.exists(task_id=task_id, path=plan_artifact_name):
                plan_data = self.workspace.read(task_id=task_id, path=plan_artifact_name)
                plan = json.loads(plan_data.decode())
                LOG.info(f"Loaded existing plan: {pprint.pformat(plan)}")
            else:
                LOG.info(f"No existing plan found for task {task_id}. Orchestrator is invoking PlannerAgent.")

                # Instantiate PlannerAgent - This assumes direct class instantiation is feasible.
                # In a distributed MAS, this would be a service call (e.g., via Agent Protocol).
                # We pass the same db and workspace for now; this might need refinement
                # if PlannerAgent needs its own isolated workspace for its internal artifacts.
                try:
                    # Adjust import path as necessary. This might need sys.path manipulation
                    # or a proper installation of PlannerAgent as a package if run in separate processes.
                    # For now, assume it's discoverable.
                    from autogpts.PlannerAgent.forge.agent import ForgeAgent as PlannerForgeAgent

                    planner_agent = PlannerForgeAgent(self.db, self.workspace) # Or a dedicated workspace for planner
                    LOG.info(f"Orchestrator: Instantiated PlannerAgent. User goal: {user_goal}")

                    # Call a method on PlannerAgent to generate the plan.
                    # The PlannerAgent's execute_step itself saves its plan as an artifact.
                    # Here, we need the plan data returned to the Orchestrator.
                    # So, PlannerAgent's generate_plan_from_goal method is more suitable.
                    generated_plan_from_planner = await planner_agent.generate_plan_from_goal(user_goal, task_id, step.step_id)

                    if not generated_plan_from_planner or (len(generated_plan_from_planner) == 1 and generated_plan_from_planner[0]['status'] == 'failed'):
                        LOG.error("PlannerAgent failed to generate a valid plan.")
                        step.output = "Error: PlannerAgent failed to generate a plan."
                        step.is_last = True
                        return step

                    plan = generated_plan_from_planner
                    LOG.info(f"Orchestrator: Received plan from PlannerAgent: {pprint.pformat(plan)}")

                except ImportError as ie:
                    LOG.error(f"Failed to import PlannerAgent: {ie}. Ensure it's correctly structured and accessible.")
                    LOG.error("Falling back to static plan generation for Orchestrator.")
                    # Fallback to static plan if PlannerAgent cannot be imported/used
                    plan = [
                        {"task_id": "static-plan-1", "description": "Static: Decompose user goal.", "agent_role": "PlannerAgent", "status": "pending", "dependencies": [], "output": ""},
                        {"task_id": "static-research-1", "description": "Static: Gather info.", "agent_role": "ResearcherAgent", "status": "pending", "dependencies": ["static-plan-1"], "output": ""},
                    ]
                except Exception as planner_exc:
                    LOG.error(f"Error during PlannerAgent invocation or plan generation: {planner_exc}")
                    LOG.error("Falling back to static plan generation for Orchestrator.")
                    # Fallback to static plan
                    plan = [
                        {"task_id": "static-plan-error-1", "description": "Static (error fallback): Decompose goal.", "agent_role": "PlannerAgent", "status": "pending", "dependencies": [], "output": ""},
                    ]

                # Save the plan received from PlannerAgent (or fallback) into Orchestrator's workspace
                self.workspace.write(task_id=task_id, path=plan_artifact_name, data=json.dumps(plan, indent=4).encode())
                await self.db.create_artifact(task_id=task_id, step_id=step.step_id, file_name=plan_artifact_name, relative_path="", agent_created=True) # Orchestrator created this artifact from Planner's output
                LOG.info(f"Orchestrator saved plan (from PlannerAgent or fallback) as {plan_artifact_name}: {pprint.pformat(plan)}")

                # Save user_goal as an artifact if this is the first planning step
                user_goal_artifact_name = "user_goal.txt"
                if not self.workspace.exists(task_id=task_id, path=user_goal_artifact_name):
                    self.workspace.write(task_id=task_id, path=user_goal_artifact_name, data=user_goal.encode())
                    await self.db.create_artifact(task_id=task_id, step_id=step.step_id, file_name=user_goal_artifact_name, relative_path="", agent_created=True)
                    LOG.info(f"Orchestrator saved user_goal.txt for task {task_id}")

                step_output_message = f"Orchestrator received plan from PlannerAgent (or fallback) for goal: '{user_goal}'. Plan stored as {plan_artifact_name}. Next step will be to execute the first task from this plan."
                step.output = json.dumps({
                    "message": step_output_message,
                    "next_actionable_task_id": plan[0]["task_id"] if plan and plan[0]["status"] == "pending" else None, # Assuming first task is next if plan just created
                    "last_completed_task_id": None,
                    "last_completed_task_artifact": None,
                    "plan_status_summary": self._get_plan_status_summary(plan),
                    "overall_status": "PLANNING_COMPLETE"
                })
                step.is_last = False # More steps to follow to execute the plan
                return step

        except Exception as e:
            LOG.error(f"Error loading or generating plan in Orchestrator: {e}")
            step.output = json.dumps({
                "message": f"Error in Orchestrator's planning phase: {e}",
                "next_actionable_task_id": None,
                "last_completed_task_id": None,
                "last_completed_task_artifact": None,
                "plan_status_summary": self._get_plan_status_summary([]), # Empty plan on error
                "overall_status": "ERROR_PLANNING"
            })
            step.is_last = True
            return step

        # --- Centralized plan saving function ---
        # This ensures plan is saved whenever its state might change.
        # However, in the current flow, it's mostly saved after a task is processed or at the end.
        # The initial save is handled above.

        # 2. Identify Next Task
        next_task_to_execute = None
        for i, planned_task in enumerate(plan):
            if planned_task["status"] == "pending":
                # Check dependencies
                dependencies_met = True
                if planned_task["dependencies"]:
                    for dep_id in planned_task["dependencies"]:
                        dependency_task = next((t for t in plan if t["task_id"] == dep_id), None)
                        if not dependency_task or dependency_task["status"] != "completed":
                            dependencies_met = False
                            break
                if dependencies_met:
                    next_task_to_execute = planned_task
                    plan[i]["status"] = "in_progress" # Mark as in_progress
                    break


        message = ""
        next_actionable_task_id = None
        last_completed_task_id = None
        last_completed_task_artifact = None
        overall_status = "EXECUTING_PLAN"


        if next_task_to_execute:
            LOG.info(f"Next task to execute: {next_task_to_execute['task_id']} - {next_task_to_execute['description']}")

            # 3. Simulate Task Delegation
            delegation_result = None
            agent_role = next_task_to_execute["agent_role"]

            if agent_role == "PlannerAgent": # Should ideally not happen again if plan exists
                delegation_result = await self._delegate_to_planner(task_id, step.step_id, next_task_to_execute)
            elif agent_role == "ResearcherAgent":
                delegation_result = await self._delegate_to_researcher(task_id, step.step_id, next_task_to_execute, user_goal)
            elif agent_role == "CoderAgent":
                delegation_result = await self._delegate_to_coder(task_id, step.step_id, next_task_to_execute, user_goal)
            elif agent_role == "ReviewerAgent":
                delegation_result = await self._delegate_to_reviewer(task_id, step.step_id, next_task_to_execute, user_goal, plan)
            elif agent_role == "OrchestratorAgent": # Assuming Orchestrator handles finalization
                delegation_result = await self._finalize_task(task_id, step.step_id, next_task_to_execute)
            else:
                LOG.error(f"Unknown agent role: {agent_role} for task {next_task_to_execute['task_id']}")
                message = f"Error: Unknown agent role '{agent_role}' for task {next_task_to_execute['task_id']}."
                # Update plan status for the failed task
                for i_err, err_task in enumerate(plan):
                    if err_task["task_id"] == next_task_to_execute["task_id"]:
                        plan[i_err]["status"] = "failed"
                        break
                overall_status = "ERROR_IN_EXECUTION"


            # 4. Update Plan Status
            if delegation_result:
                last_completed_task_id = next_task_to_execute["task_id"]
                last_completed_task_artifact = delegation_result.get("output_artifact", "")

                for i_update, pt_task_update in enumerate(plan):
                    if pt_task_update["task_id"] == last_completed_task_id:
                        plan[i_update]["status"] = "completed"
                        plan[i_update]["output"] = last_completed_task_artifact
                        LOG.info(f"Task '{last_completed_task_id}' marked as completed. Output: {last_completed_task_artifact}")
                        break
                message = f"Task '{next_task_to_execute['description']}' ({agent_role}) was delegated and completed. Output: {last_completed_task_artifact}."
            elif not message: # If no error message was set from unknown role
                 message = f"Task '{next_task_to_execute['description']}' ({agent_role}) failed or no result."
                 # Ensure status is updated if delegation_result is None but no specific error message set
                 for i_fail, fail_task in enumerate(plan):
                    if fail_task["task_id"] == next_task_to_execute["task_id"] and fail_task["status"] == "in_progress":
                        plan[i_fail]["status"] = "failed" # Mark as failed if delegation didn't complete successfully
                        break


            # 5. Save Updated Plan Consistently
            try:
                self.workspace.write(task_id=task_id, path=plan_artifact_name, data=json.dumps(plan, indent=4).encode())
                LOG.info(f"Updated plan saved to {plan_artifact_name} after processing {last_completed_task_id or 'task'}")
            except Exception as e:
                LOG.error(f"Error saving updated plan: {e}")
                message += f" CRITICAL: Error saving plan: {e}"
                overall_status = "ERROR_SAVING_PLAN"

            # Determine next actionable task for the message and overall status
            current_next_actionable_task = None
            for t_next in plan:
                if t_next["status"] == "pending":
                    deps_met = True
                    if t_next["dependencies"]:
                        for dep_id_next in t_next["dependencies"]:
                            dep_task_next = next((pt_next for pt_next in plan if pt_next["task_id"] == dep_id_next), None)
                            if not dep_task_next or dep_task_next["status"] != "completed":
                                deps_met = False
                                break
                    if deps_met:
                        current_next_actionable_task = t_next
                        break

            if current_next_actionable_task:
                next_actionable_task_id = current_next_actionable_task["task_id"]
                message += f"\nNext up: '{current_next_actionable_task['description']}' ({current_next_actionable_task['agent_role']})."
                step.is_last = False
            else: # No more actionable tasks
                if all(t["status"] == "completed" for t in plan):
                    message += f"\nAll tasks in the plan are completed. User goal '{user_goal}' is finished."
                    overall_status = "ALL_TASKS_COMPLETE"
                    step.is_last = True
                elif any(t["status"] == "failed" for t in plan):
                    overall_status = "ERROR_IN_EXECUTION"
                    message += "\nPlan execution cannot continue due to failed tasks."
                    step.is_last = True
                else: # Pending tasks but dependencies not met
                    overall_status = "BLOCKED"
                    message += "\nNo further tasks can be processed at this moment (waiting for dependencies)."
                    step.is_last = True # Or False if we want periodic re-evaluation by calling agent again

        else: # No task was identified to execute in this step
            if all(t["status"] == "completed" for t in plan):
                message = f"All tasks in the plan are already completed. User goal '{user_goal}' is finished."
                overall_status = "ALL_TASKS_COMPLETE"
                step.is_last = True
            elif any(t["status"] == "failed" for t in plan):
                overall_status = "ERROR_IN_EXECUTION"
                message = "\nPlan execution cannot continue due to failed tasks (checked at start of step)."
                step.is_last = True
            else:
                message = "No actionable tasks found (either all done, blocked by dependencies, or plan is empty)."
                overall_status = "BLOCKED" if any(t["status"] == "pending" for t in plan) else "UNKNOWN_STATE"
                step.is_last = True

        step.output = json.dumps({
            "message": message,
            "next_actionable_task_id": next_actionable_task_id,
            "last_completed_task_id": last_completed_task_id,
            "last_completed_task_artifact": last_completed_task_artifact,
            "plan_status_summary": self._get_plan_status_summary(plan),
            "overall_status": overall_status
        })
        LOG.info(f"\t✅ Step {step.step_id} completed. Output: {step.output}")
        return step

    def _get_plan_status_summary(self, plan: list) -> dict:
        summary = {"total_tasks": len(plan), "pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for task in plan:
            status = task.get("status", "unknown")
            if status in summary:
                summary[status] += 1
            else:
                summary[status] = 1 # For any other statuses
        return summary
