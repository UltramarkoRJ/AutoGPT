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
from enum import Enum

LOG = ForgeLogger(__name__)

class DVMode(Enum):
    AWAITING_GOAL = "AWAITING_GOAL"
    AWAITING_USER_CLARIFICATIONS = "AWAITING_USER_CLARIFICATIONS"
    AWAITING_REFINED_GOAL_APPROVAL = "AWAITING_REFINED_GOAL_APPROVAL"
    RESOURCE_PLANNING = "RESOURCE_PLANNING"
    AWAITING_RESOURCE_PLAN_APPROVAL = "AWAITING_RESOURCE_PLAN_APPROVAL"
    TEAM_ASSEMBLY = "TEAM_ASSEMBLY"
    AWAITING_TEAM_PROPOSAL_APPROVAL = "AWAITING_TEAM_PROPOSAL_APPROVAL"
    ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD = "ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD"
    ORCHESTRATING_INITIATED = "ORCHESTRATING_INITIATED"
    ORCHESTRATING_TASKS = "ORCHESTRATING_TASKS" # Represents ongoing execution
    PROJECT_COMPLETED = "PROJECT_COMPLETED"
    PROJECT_FAILED = "PROJECT_FAILED"

class DiretoraVirtualForgeAgent(Agent):
    """
    The Diretora Virtual (DV) Agent orchestrates complex user goals by
    clarifying objectives, planning resources (virtual companies),
    assembling teams of specialized agents, and overseeing execution.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        self.dv_mode = DVMode.AWAITING_GOAL
        self.user_goal_initial = None
        self.user_goal_refined = None
        self.generated_questions = []
        self.user_answers_to_questions = {}
        self.conversation_history = []
        self.pending_approval_type = None
        self.proposed_structure = None
        self.approved_structure = None
        self.team_proposals = {}
        self.current_company_index_for_assembly = 0
        # Orchestration specific state
        self.current_orchestration_company_index = 0
        self.current_orchestration_task_index = 0 # Within a company's plan (conceptual)
        self.orchestration_status_message = ""

        self.prompt_engine = PromptEngine("gpt-4-turbo-preview")
        LOG.info("DiretoraVirtualAgent initialized in AWAITING_GOAL mode.")

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # The main task for DV is the overarching user goal.
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 DiretoraVirtualAgent Task created: {task.task_id} input: {task.input[:60]}{'...' if len(task.input) > 60 else ''}"
        )
        # If the task input is the initial goal, process it immediately.
        if self.dv_mode == DVMode.AWAITING_GOAL and task.input:
            self.user_goal_initial = task.input
            self.conversation_history.append({"role": "user", "content": f"Initial Goal: {self.user_goal_initial}"})
            LOG.info(f"Initial user goal received: {self.user_goal_initial}")
        return task

    def _create_goal_analysis_prompt(self, initial_goal: str) -> str:
        return f"""
        You are the "Diretora Virtual," an AI project director. Your first task is to analyze the user's initial goal and identify areas that need clarification to ensure a successful project outcome.

        User's Initial Goal: "{initial_goal}"

        Based on this goal, please generate 3-5 strategic clarifying questions. These questions should help to:
        1. Define the scope more precisely.
        2. Identify key deliverables.
        3. Understand constraints (e.g., budget, time, technology preferences).
        4. Determine success criteria.
        5. Uncover any hidden assumptions.

        Return ONLY a JSON list of strings, where each string is a question. Do not include any other text or explanations.
        Example: ["What is the primary expected outcome of this project?", "Are there any specific deadlines to consider?", "What does success look like for this project?"]
        """

    def _create_refined_goal_prompt(self, initial_goal: str, conversation_history: list) -> str:
        conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        return f"""
        You are the "Diretora Virtual," an AI project director. You have received an initial goal from the user and have asked clarifying questions. Now, you need to synthesize this information into a single, well-defined, and actionable refined goal.

        Initial User Goal: "{initial_goal}"
        Conversation History (including user's answers to your questions):
        {conversation_str}

        Based on the entire conversation, synthesize a refined project goal. This refined goal should be:
        - Clear and unambiguous.
        - Concise, yet comprehensive enough to guide project planning.
        - Actionable for a team of AI agents.
        - Reflect all critical information provided by the user.

        Also, provide a brief confidence level assessment for this refined goal (e.g., "High", "Medium", "Low") based on how well-defined you believe it is now.
        Return the output as a JSON object with two keys: "refined_goal_text" and "confidence_level".
        Example: {{"refined_goal_text": "Develop a Python web application for e-commerce with features X, Y, and Z, deployed on AWS by Q4.", "confidence_level": "High"}}
        """

    def _create_resource_plan_prompt(self, refined_goal: str) -> str:
        return f"""
        You are the "Diretora Virtual," an expert AI project strategist.
        The confirmed project goal is: "{refined_goal}"

        To achieve this goal, propose a high-level structure of "virtual companies" (functional units) and potentially "sub-companies" (specialized teams within a virtual company).
        For each virtual company and sub-company, define:
        1.  `id`: A unique string identifier (e.g., "vc_data_analysis", "sc_web_frontend_team").
        2.  `name`: A concise, descriptive name (e.g., "Data Analytics Unit", "Web Frontend Team").
        3.  `type`: Either "virtual_company" or "sub_company".
        4.  `parent_company_id`: The ID of the parent virtual company if this is a sub-company, otherwise null.
        5.  `purpose`: A clear statement of what this unit is responsible for delivering or achieving.
        6.  `rationale`: A brief explanation of why this unit is necessary to achieve the overall project goal.

        Also, provide an `overall_project_rationale` explaining why this overall structure is optimal for the goal.
        The primary focus is on defining the top-level virtual companies. Sub-companies are optional and should only be defined if a virtual company has clearly distinct internal specializations that need to be highlighted at this stage.

        Return the output as a single JSON object with two keys:
        - `overall_project_rationale` (string)
        - `structure_plan` (a list of company/sub-company objects as defined above).
        Ensure the response is ONLY the JSON object.
        """

    def get_mock_agent_pool(self) -> list:
        # Simplified agent manifests for now
        return [
            {
                "agentTypeId": "com.autogpt.agents.coder.python.v1",
                "displayName": "Python Coder Agent",
                "description": "Specializes in writing, debugging, and explaining Python code.",
                "tags": ["python", "coding", "development", "backend"],
                "capabilities_summary": "generate_python_code, debug_python_code, implement_algorithms"
            },
            {
                "agentTypeId": "com.autogpt.agents.researcher.web.v1",
                "displayName": "Web Researcher Agent",
                "description": "Gathers information from the web, summarizes articles, and finds data.",
                "tags": ["research", "web", "data_collection", "analysis"],
                "capabilities_summary": "perform_web_search, summarize_text, extract_information_from_url"
            },
            {
                "agentTypeId": "com.autogpt.agents.planner.project.v1",
                "displayName": "Project Planner Agent",
                "description": "Breaks down complex goals into detailed task plans.",
                "tags": ["planning", "project_management", "task_decomposition"],
                "capabilities_summary": "create_task_plan, identify_dependencies, estimate_task_duration"
            },
            {
                "agentTypeId": "com.autogpt.agents.reviewer.code.v1",
                "displayName": "Code Reviewer Agent",
                "description": "Reviews code for quality, adherence to standards, and potential bugs.",
                "tags": ["code_review", "quality_assurance", "testing"],
                "capabilities_summary": "analyze_code_quality, check_coding_standards, identify_bugs"
            }
        ]

    def _create_team_assembly_prompt(self, company_details: dict, refined_goal: str, agent_pool_manifests: list) -> str:
        company_name = company_details.get("name", "N/A")
        company_purpose = company_details.get("purpose", "N/A")

        agent_pool_description = "\n".join([
            f"- ID: {agent['agentTypeId']}, Name: {agent['displayName']}, Description: {agent['description']}, Capabilities: {agent['capabilities_summary']}, Tags: {', '.join(agent['tags'])}"
            for agent in agent_pool_manifests
        ])

        return f"""
        You are the "Diretora Virtual," an expert AI team designer.
        The overall project goal is: "{refined_goal}"
        You are currently assembling a team for the virtual company named "{company_name}".
        This company's specific purpose is: "{company_purpose}".

        Available Agent Pool (Agent Types):
        {agent_pool_description}

        Based on the company's purpose and the overall project goal:
        1. Define 1-3 key agent roles needed for this company.
        2. For each role, specify:
            - `role_in_company`: A descriptive name for the role (e.g., "Lead Developer", "Data Gatherer", "API Integrator").
            - `role_description`: What this role is responsible for within this company.
            - `required_skills_summary`: A brief summary of skills needed (e.g., "Python API development", "Web research and summarization").
            - `selected_agentTypeId`: The `agentTypeId` from the Available Agent Pool that is the best fit for this role.
            - `justification_for_selection`: A brief reason why this agent type was selected for this role.
        3. Provide a `collaboration_summary` (1-2 sentences) on how these selected agents are expected to collaborate within this company to achieve its purpose.

        Return ONLY a JSON object structured as follows:
        {{
          "company_id": "{company_details.get('id')}",
          "company_name": "{company_name}",
          "proposed_team": [
            // list of agent role objects as defined above
          ],
          "collaboration_summary": "..."
        }}
        Example for one agent in proposed_team:
        {{
            "role_in_company": "Lead Python Developer",
            "role_description": "Develops and maintains the core Python application logic.",
            "required_skills_summary": "Advanced Python, API design, database management.",
            "selected_agentTypeId": "com.autogpt.agents.coder.python.v1",
            "justification_for_selection": "This agent specializes in Python coding and matches the skill requirements."
        }}
        """

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        LOG.info(f"DiretoraVirtualAgent - Current Mode: {self.dv_mode.value}")
        LOG.info(f"Input for step: {step_request.input[:100] if step_request.input else 'No input provided in step_request'}")
        step = await self.db.create_step(task_id=task_id, input=step_request)

        # Phase 1: Initial Interaction
        if self.dv_mode == DVMode.AWAITING_GOAL:
            if not step_request.input:
                LOG.warning("No input provided in AWAITING_GOAL mode.")
                step.output = json.dumps({"status": "ERROR", "message": "No goal provided. Please provide your project goal."})
                step.is_last = True
                return step

            self.user_goal_initial = step_request.input
            self.conversation_history.append({"role": "user", "content": f"Initial Goal: {self.user_goal_initial}"})
            LOG.info(f"Received initial user goal: {self.user_goal_initial}")

            prompt_text = self._create_goal_analysis_prompt(self.user_goal_initial)
            try:
                response = await chat_completion_request(
                    messages=[{"role": "system", "content": "You are an AI assistant that generates clarifying questions as a JSON list."},
                              {"role": "user", "content": prompt_text}],
                    model=self.prompt_engine.model_name
                )
                questions_str = response["choices"][0]["message"]["content"]
                self.generated_questions = json.loads(questions_str)
                self.conversation_history.append({"role": "assistant", "content": f"Generated clarifying questions: {self.generated_questions}"})

                self.dv_mode = DVMode.AWAITING_USER_CLARIFICATIONS
                self.pending_approval_type = "clarification_answers"
                step.output = json.dumps({
                    "status": "USER_INPUT_REQUIRED",
                    "message": "Please answer the following questions to clarify your goal:",
                    "questions": self.generated_questions,
                    "pending_approval_type": self.pending_approval_type
                })
                LOG.info(f"Generated questions for user: {self.generated_questions}")
            except Exception as e:
                LOG.error(f"Error generating clarifying questions: {e}")
                step.output = json.dumps({"status": "ERROR", "message": f"Failed to generate clarifying questions: {e}"})
            step.is_last = True

        elif self.dv_mode == DVMode.AWAITING_USER_CLARIFICATIONS:
            if not step_request.input:
                LOG.warning("No input (answers) provided in AWAITING_USER_CLARIFICATIONS mode.")
                step.output = json.dumps({
                    "status": "USER_INPUT_REQUIRED",
                    "message": "No answers received. Please answer the previous questions:",
                    "questions": self.generated_questions, # Re-send questions
                    "pending_approval_type": "clarification_answers"
                })
                step.is_last = True
                return step

            user_answers_text = step_request.input
            self.user_answers_to_questions = {"summary_of_answers": user_answers_text}
            self.conversation_history.append({"role": "user", "content": f"User Answers: {user_answers_text}"})
            LOG.info(f"Received user answers: {user_answers_text[:200]}")

            prompt_text = self._create_refined_goal_prompt(self.user_goal_initial, self.conversation_history)
            try:
                response = await chat_completion_request(
                    messages=[{"role": "system", "content": "You are an AI assistant that synthesizes goals into a JSON object."},
                              {"role": "user", "content": prompt_text}],
                    model=self.prompt_engine.model_name
                )
                refined_goal_data_str = response["choices"][0]["message"]["content"]
                refined_goal_data = json.loads(refined_goal_data_str)

                self.user_goal_refined = refined_goal_data.get("refined_goal_text")
                confidence = refined_goal_data.get("confidence_level", "N/A")
                self.conversation_history.append({"role": "assistant", "content": f"Proposed Refined Goal: {self.user_goal_refined} (Confidence: {confidence})"})

                self.dv_mode = DVMode.AWAITING_REFINED_GOAL_APPROVAL
                self.pending_approval_type = "refined_goal_approval"
                step.output = json.dumps({
                    "status": "USER_APPROVAL_REQUIRED",
                    "message": f"Based on your answers, my understanding of your goal is now: \"{self.user_goal_refined}\" (Confidence: {confidence}). Do you approve this refined goal to proceed with resource planning?",
                    "refined_goal": self.user_goal_refined,
                    "confidence": confidence,
                    "pending_approval_type": self.pending_approval_type
                })
                LOG.info(f"Proposed refined goal: {self.user_goal_refined} (Confidence: {confidence})")
            except Exception as e:
                LOG.error(f"Error generating refined goal: {e}")
                step.output = json.dumps({"status": "ERROR", "message": f"Failed to generate refined goal: {e}"})
                self.dv_mode = DVMode.AWAITING_USER_CLARIFICATIONS
            step.is_last = True

        elif self.dv_mode == DVMode.AWAITING_REFINED_GOAL_APPROVAL:
            # Assuming step_request.input is a simple "yes" or "no" or a JSON like {"approved": true/false}
            approval_response = step_request.input.lower() if isinstance(step_request.input, str) else ""
            approved = False
            if approval_response == "yes": # Basic approval check
                approved = True
            try: # More robust check for JSON input
                approval_data = json.loads(step_request.input)
                if isinstance(approval_data, dict) and approval_data.get("approved") == True:
                    approved = True
            except (json.JSONDecodeError, TypeError):
                 pass # Keep `approved` as determined by string check or default False

            self.conversation_history.append({"role": "user", "content": f"Approval for refined goal: {step_request.input}"})

            if approved:
                LOG.info(f"Refined goal approved by user: {self.user_goal_refined}")
                self.dv_mode = DVMode.RESOURCE_PLANNING
                self.pending_approval_type = None
                step.output = json.dumps({
                    "status": "PROCESSING", # Or "TRANSITIONING"
                    "message": "Refined goal approved. Proceeding to resource planning..."
                })
                step.is_last = False # To allow immediate execution of RESOURCE_PLANNING if possible
            else:
                LOG.info("Refined goal not approved by user.")
                self.conversation_history.append({"role": "assistant", "content": "User did not approve the refined goal. Restarting or awaiting new input."})
                self.dv_mode = DVMode.AWAITING_GOAL # Revert to awaiting a new or modified goal
                self.pending_approval_type = None # Reset
                self.user_goal_initial = None # Clear previous goal attempt
                self.user_goal_refined = None
                self.generated_questions = []
                self.user_answers_to_questions = {}
                step.output = json.dumps({
                    "status": "HALTED_AWAITING_NEW_INPUT",
                    "message": "Refined goal not approved. Please provide a new or modified project goal if you wish to continue.",
                    "pending_approval_type": None
                })
                step.is_last = True

        # Phase 2: Resource Planning (will execute if mode changed to RESOURCE_PLANNING and step.is_last=False)
        # This structure allows chaining of logic within a single execute_step call if appropriate
        if self.dv_mode == DVMode.RESOURCE_PLANNING and not step.is_last: # Check is_last in case previous phase decided to end step
            LOG.info("Transitioning to Resource Planning phase.")
            if not self.user_goal_refined:
                LOG.error("Cannot proceed to resource planning without a refined goal.")
                step.output = json.dumps({"status": "ERROR", "message": "Cannot plan resources: Refined goal is missing."})
                self.dv_mode = DVMode.AWAITING_GOAL # Revert
                step.is_last = True
                return step

            prompt_text = self._create_resource_plan_prompt(self.user_goal_refined)
            try:
                response = await chat_completion_request(
                     messages=[{"role": "system", "content": "You are an AI assistant that designs project structures as a JSON object."},
                               {"role": "user", "content": prompt_text}],
                     model=self.prompt_engine.model_name
                )
                resource_plan_str = response["choices"][0]["message"]["content"]
                self.proposed_structure = json.loads(resource_plan_str) # Store the raw dict
                self.conversation_history.append({"role": "assistant", "content": f"Proposed Resource Plan: {pprint.pformat(self.proposed_structure)}"})

                self.dv_mode = DVMode.AWAITING_RESOURCE_PLAN_APPROVAL
                self.pending_approval_type = "resource_plan_approval"
                step.output = json.dumps({
                    "status": "USER_APPROVAL_REQUIRED",
                    "message": "I have drafted a resource plan. Please review the proposed virtual company structure.",
                    "resource_plan": self.proposed_structure,
                    "pending_approval_type": self.pending_approval_type
                })
                LOG.info(f"Proposed resource plan generated: {pprint.pformat(self.proposed_structure)}")
            except Exception as e:
                LOG.error(f"Error generating resource plan: {e}")
                step.output = json.dumps({"status": "ERROR", "message": f"Failed to generate resource plan: {e}"})
                self.dv_mode = DVMode.RESOURCE_PLANNING
            step.is_last = True

        elif self.dv_mode == DVMode.AWAITING_RESOURCE_PLAN_APPROVAL:
            approval_response = step_request.input.lower() if isinstance(step_request.input, str) else ""
            approved = False
            if approval_response == "yes":
                approved = True
            try:
                approval_data = json.loads(step_request.input)
                if isinstance(approval_data, dict) and approval_data.get("approved") == True:
                    approved = True
            except (json.JSONDecodeError, TypeError):
                 pass

            self.conversation_history.append({"role": "user", "content": f"Approval for resource plan: {step_request.input}"})

            if approved:
                LOG.info(f"Resource plan approved by user: {pprint.pformat(self.proposed_structure)}")
                    # Initialize approved_structure with space for teams
                    self.approved_structure = json.loads(json.dumps(self.proposed_structure)) # Deep copy
                    if "structure_plan" in self.approved_structure: # Ensure key exists
                        for company in self.approved_structure["structure_plan"]:
                            company["approved_team"] = None # Initialize field for approved team
                            company["team_approval_status"] = "pending" # 'pending', 'approved', 'rejected'

                self.dv_mode = DVMode.TEAM_ASSEMBLY
                self.pending_approval_type = None
                    self.team_proposals = {}
                    self.current_company_index_for_assembly = 0
                step.output = json.dumps({
                    "status": "PROCESSING",
                    "message": "Resource plan approved. Proceeding to team assembly..."
                })
                    step.is_last = False
            else:
                LOG.info("Resource plan not approved by user.")
                    self.dv_mode = DVMode.AWAITING_RESOURCE_PLAN_APPROVAL
                step.output = json.dumps({
                        "status": "HALTED_AWAITING_NEW_INPUT",
                    "message": "Resource plan not approved. Please provide feedback or suggest modifications to the plan. For now, you might need to restart the goal or provide a modified resource plan if the system supported it.",
                        "pending_approval_type": "resource_plan_feedback"
                })
                step.is_last = True

        # Phase 3: Team Assembly (iterates through companies)
        if self.dv_mode == DVMode.TEAM_ASSEMBLY and not step.is_last:
            LOG.info(f"Starting/Continuing Team Assembly. Current company index: {self.current_company_index_for_assembly}")
            if not self.approved_structure or not self.approved_structure.get("structure_plan"):
                LOG.error("Cannot proceed to team assembly without an approved resource structure.")
                step.output = json.dumps({"status": "ERROR", "message": "Cannot assemble teams: Approved resource structure is missing."})
                self.dv_mode = DVMode.RESOURCE_PLANNING
                step.is_last = True
                return step

            companies = self.approved_structure.get("structure_plan", [])
            if self.current_company_index_for_assembly < len(companies):
                current_company = companies[self.current_company_index_for_assembly]

                # Skip if team already approved for this company (e.g. if user rejected a later team and we looped back)
                if current_company.get("team_approval_status") == "approved":
                    LOG.info(f"Team for company '{current_company.get('name')}' already approved. Skipping to next.")
                    self.current_company_index_for_assembly += 1
                    # This recursive call or loop structure needs careful handling to avoid deep stacks if many are skipped.
                    # For now, let's assume it will re-enter TEAM_ASSEMBLY mode in the next step if not is_last.
                    # To make it continue in the same step:
                    step.is_last = False # Signal to re-evaluate TEAM_ASSEMBLY in this same step execution
                    # Re-evaluate TEAM_ASSEMBLY immediately by creating a new "step" in the logic, not a new Agent Protocol step
                    # This is a bit of a hack for immediate re-looping. A cleaner way might be a while loop.
                    # For now, let's assume the next call to execute_step will pick it up.
                    # To force it, we'd need to structure this part of execute_step as a loop or recursive call.
                    # For simplicity, this will be handled by the next step call.
                    # If we want it to immediately proceed:
                    # return await self.execute_step(task_id, step_request) # Risky due to potential loops without state change.
                    # Best to let the next step handle it.
                    # The output should reflect that we are skipping or just that we are processing.
                    # If we set is_last=False, the message might be overwritten.
                    # Let's ensure this step has a meaningful output before potential re-loop.
                    # The logic below will handle the next company or transition to ALL_TEAMS_ASSEMBLED.
                    # This means if a company is skipped, the next one will be processed, or it will transition.
                    # This is handled by the subsequent check for current_company_index_for_assembly < len(companies)

                else: # Team not yet approved, so propose it
                    LOG.info(f"Assembling team for company: {current_company.get('name')}")
                    mock_agent_pool = self.get_mock_agent_pool()
                    prompt_text = self._create_team_assembly_prompt(current_company, self.user_goal_refined, mock_agent_pool)

                    try:
                        response = await chat_completion_request(
                            messages=[{"role": "system", "content": "You are an AI assistant that designs teams as a JSON object."},
                                      {"role": "user", "content": prompt_text}],
                            model=self.prompt_engine.model_name
                        )
                        team_proposal_str = response["choices"][0]["message"]["content"]
                        proposed_team_data = json.loads(team_proposal_str)

                        self.team_proposals[current_company['id']] = proposed_team_data # Store the latest proposal
                        self.conversation_history.append({"role": "assistant", "content": f"Proposed team for {current_company['name']}: {pprint.pformat(proposed_team_data)}"})

                        self.dv_mode = DVMode.AWAITING_TEAM_PROPOSAL_APPROVAL
                        self.pending_approval_type = f"team_proposal_approval_for_company_{current_company['id']}"
                        step.output = json.dumps({
                            "status": "USER_APPROVAL_REQUIRED",
                            "message": f"Please review the proposed team for company '{current_company['name']}'.",
                            "company_id": current_company['id'],
                            "company_name": current_company['name'],
                            "proposed_team_details": proposed_team_data,
                            "pending_approval_type": self.pending_approval_type
                        })
                        LOG.info(f"Proposed team for company '{current_company['name']}': {pprint.pformat(proposed_team_data)}")
                    except Exception as e:
                        LOG.error(f"Error generating team proposal for {current_company['name']}: {e}")
                        step.output = json.dumps({"status": "ERROR", "message": f"Failed to generate team proposal for {current_company['name']}: {e}"})
                    step.is_last = True # Wait for approval for this company's team
                    return step # Return here as we are awaiting approval for this specific team

            # This 'else' corresponds to: if self.current_company_index_for_assembly < len(companies)
            # If the index is now >= len(companies), all companies have been processed.
            LOG.info("All companies have had teams proposed and iteratively approved.")
            self.dv_mode = DVMode.ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD
            self.pending_approval_type = "confirmation_to_start_orchestration"
            # Prepare a summary of the approved structure with teams for the user
            final_structure_summary = json.loads(json.dumps(self.approved_structure)) # Deep copy
            for company_summary in final_structure_summary.get("structure_plan", []):
                # Ensure 'approved_team' is populated from self.team_proposals if it was just set
                # This part might be redundant if self.approved_structure is updated directly
                if company_summary.get("team_approval_status") == "approved" and not company_summary.get("approved_team"):
                     company_summary["approved_team"] = self.team_proposals.get(company_summary["id"], {}).get("proposed_team", "Error retrieving team")

            step.output = json.dumps({
                "status": "USER_APPROVAL_REQUIRED",
                "message": "All teams have been assembled and approved. Ready to start orchestration. Do you want to proceed?",
                "pending_approval_type": self.pending_approval_type,
                "final_team_structure": final_structure_summary
            })
            step.is_last = True

        elif self.dv_mode == DVMode.AWAITING_TEAM_PROPOSAL_APPROVAL:
            # Extract company_id from pending_approval_type
            # e.g., "team_proposal_approval_for_company_vc_001"
            if not self.pending_approval_type or not self.pending_approval_type.startswith("team_proposal_approval_for_company_"):
                LOG.error(f"Invalid pending_approval_type for AWAITING_TEAM_PROPOSAL_APPROVAL: {self.pending_approval_type}")
                step.output = json.dumps({"status": "ERROR", "message": "Internal error: Invalid pending approval type."})
                step.is_last = True
                return step

            company_id_for_approval = self.pending_approval_type.replace("team_proposal_approval_for_company_", "")

            approval_response = step_request.input.lower() if isinstance(step_request.input, str) else ""
            approved = False
            if approval_response == "yes":
                approved = True
            try:
                approval_data = json.loads(step_request.input)
                if isinstance(approval_data, dict) and approval_data.get("approved") == True:
                    approved = True
            except (json.JSONDecodeError, TypeError):
                pass

            self.conversation_history.append({"role": "user", "content": f"Approval for team of company {company_id_for_approval}: {step_request.input}"})

            company_to_update = None
            company_idx = -1
            for idx, company in enumerate(self.approved_structure.get("structure_plan", [])):
                if company["id"] == company_id_for_approval:
                    company_to_update = company
                    company_idx = idx
                    break

            if not company_to_update:
                LOG.error(f"Could not find company {company_id_for_approval} in approved_structure to update team status.")
                step.output = json.dumps({"status": "ERROR", "message": f"Internal error: Company {company_id_for_approval} not found."})
                step.is_last = True
                return step

            if approved:
                LOG.info(f"Team for company '{company_to_update['name']}' approved by user.")
                company_to_update["team_approval_status"] = "approved"
                # Store the specific approved team proposal within the company's structure
                company_to_update["approved_team"] = self.team_proposals.get(company_id_for_approval, {}).get("proposed_team", "Error: Proposal not found")

                self.current_company_index_for_assembly += 1 # Move to the next company
                self.dv_mode = DVMode.TEAM_ASSEMBLY # Go back to TEAM_ASSEMBLY to process next company or finalize
                self.pending_approval_type = None
                step.output = json.dumps({
                    "status": "PROCESSING",
                    "message": f"Team for company '{company_to_update['name']}' approved. Proceeding to assemble next team or finalize team assembly."
                })
                step.is_last = False # Allow TEAM_ASSEMBLY to continue for the next company or finalize
            else:
                LOG.info(f"Team for company '{company_to_update['name']}' not approved by user.")
                company_to_update["team_approval_status"] = "rejected"
                # For now, we halt and ask for feedback. A more complex system could allow re-tries for this specific team.
                self.dv_mode = DVMode.AWAITING_TEAM_PROPOSAL_APPROVAL # Stay in this mode for this company
                step.output = json.dumps({
                    "status": "USER_INPUT_REQUIRED",
                    "message": f"Team for company '{company_to_update['name']}' not approved. Please provide feedback or suggest modifications for this team. (Currently, re-proposal for this specific team is not implemented; you might need to restart the team assembly phase or address concerns manually).",
                    "company_id": company_id_for_approval,
                    "pending_approval_type": f"team_proposal_feedback_for_company_{company_id_for_approval}" # Hypothetical
                })
                step.is_last = True


        elif self.dv_mode not in [
            DVMode.AWAITING_GOAL, DVMode.AWAITING_USER_CLARIFICATIONS,
            DVMode.AWAITING_REFINED_GOAL_APPROVAL, DVMode.RESOURCE_PLANNING,
            DVMode.AWAITING_RESOURCE_PLAN_APPROVAL, DVMode.TEAM_ASSEMBLY,
            DVMode.AWAITING_TEAM_PROPOSAL_APPROVAL, DVMode.ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD,
            DVMode.ORCHESTRATING_INITIATED # ORCHESTRATING_TASKS will be handled below or in subsequent steps
            ]:
            # Handling for ORCHESTRATING_TASKS would go here in a more complete loop
            if self.dv_mode == DVMode.ORCHESTRATING_TASKS:
                # This is where the DV would monitor ongoing tasks, handle completions,
                # and assign next tasks based on a more detailed operational plan.
                # For this subtask, ORCHESTRATING_INITIATED transitions to this,
                # and then this mode implies waiting for external events or next steps.
                step.output = json.dumps({
                    "status": "ORCHESTRATION_UNDERWAY", # Or specific task status
                    "message": self.orchestration_status_message if self.orchestration_status_message else "Orchestration is ongoing. Waiting for task completions or next trigger.",
                    # "current_task_details": { ... } // Could include details of task being monitored
                })
                LOG.info(f"Currently in ORCHESTRATING_TASKS mode. Status: {self.orchestration_status_message}")
                # is_last would depend on whether DV is actively polling or event-driven.
                # For now, assume it's waiting for next explicit step or event.
                step.is_last = True
            else:
                LOG.warning(f"Execute_step called in unhandled DVMode: {self.dv_mode}")
                step.output = json.dumps({"status": "ERROR", "message": f"Agent in unhandled mode: {self.dv_mode.value}"})
            step.is_last = True

        # Phase 2 Completion: Final "Go-Ahead" and transition to Orchestration Initiation
        if self.dv_mode == DVMode.ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD:
            approval_response = step_request.input.lower() if isinstance(step_request.input, str) else ""
            approved = False
            if approval_response == "yes":
                approved = True
            try:
                approval_data = json.loads(step_request.input)
                if isinstance(approval_data, dict) and approval_data.get("approved") == True:
                    approved = True
            except (json.JSONDecodeError, TypeError):
                pass

            self.conversation_history.append({"role": "user", "content": f"Approval to start orchestration: {step_request.input}"})

            if approved:
                LOG.info("Orchestration approved by user. Initiating orchestration phase.")
                self.dv_mode = DVMode.ORCHESTRATING_INITIATED
                self.pending_approval_type = None
                self.current_orchestration_company_index = 0 # Reset for orchestration
                self.current_orchestration_task_index = 0  # Reset for orchestration
                step.output = json.dumps({
                    "status": "PROCESSING",
                    "message": "Orchestration approved and initiated. Starting task execution..."
                })
                step.is_last = False # Allow immediate transition to ORCHESTRATING_INITIATED logic
            else:
                LOG.info("Orchestration not approved by user. System remains halted.")
                self.dv_mode = DVMode.ALL_TEAMS_ASSEMBLED_AWAITING_GO_AHEAD # Stay in this state
                step.output = json.dumps({
                    "status": "HALTED", # Or USER_INPUT_REQUIRED if we expect immediate re-prompt
                    "message": "Orchestration not approved. The system will remain in its current state with teams assembled. You can restart the orchestration approval when ready or provide new high-level goals.",
                    "pending_approval_type": "confirmation_to_start_orchestration" # Still waiting for this
                })
                step.is_last = True

        # Phase 3: Basic Orchestration Initiation (Placeholder)
        if self.dv_mode == DVMode.ORCHESTRATING_INITIATED and not step.is_last:
            LOG.info("Initiating orchestration: Assigning first conceptual task.")
            if not self.approved_structure or not self.approved_structure.get("structure_plan"):
                LOG.error("Cannot start orchestration: Approved structure with companies is missing.")
                step.output = json.dumps({"status": "ERROR", "message": "Cannot start orchestration: Critical structure information missing."})
                self.dv_mode = DVMode.AWAITING_RESOURCE_PLAN_APPROVAL # Revert to fix structure
                step.is_last = True
                return step

            first_company_list = self.approved_structure.get("structure_plan", [])
            if not first_company_list:
                LOG.error("Cannot start orchestration: No companies defined in the approved structure.")
                step.output = json.dumps({"status": "ERROR", "message": "Cannot start orchestration: No companies to orchestrate."})
                self.dv_mode = DVMode.AWAITING_RESOURCE_PLAN_APPROVAL
                step.is_last = True
                return step

            first_company = first_company_list[self.current_orchestration_company_index] # Starts at 0
            first_company_name = first_company.get("name", "N/A")

            approved_team_list = first_company.get("approved_team", [])
            if not approved_team_list:
                LOG.error(f"Cannot start orchestration for company '{first_company_name}': No approved team members.")
                step.output = json.dumps({"status": "ERROR", "message": f"No approved team for company '{first_company_name}'. Cannot assign task."})
                # This state indicates a flaw in previous approval steps, should ideally not happen.
                self.dv_mode = DVMode.TEAM_ASSEMBLY # Revert to fix team
                self.current_company_index_for_assembly = self.current_orchestration_company_index # Focus on this company
                step.is_last = True
                return step

            first_agent_details = approved_team_list[0] # Taking the first agent in the team
            first_agent_type_id = first_agent_details.get("selected_agentTypeId", "N/A")
            first_agent_role = first_agent_details.get("role_in_company", "N/A")

            conceptual_first_task_name = f"'{first_company_name}' company kickoff and readiness report"

            # Simulate Task Assignment (Placeholder for actual inter-agent communication)
            LOG.info(f"Assigning task '{conceptual_first_task_name}' to agent role '{first_agent_role}' (type: '{first_agent_type_id}') in company '{first_company_name}'.")
            # --- Future Work: Actual task assignment to the specific agent instance ---
            # This would involve:
            # 1. Retrieving/Instantiating the actual agent based on first_agent_type_id and its manifest's invocation_details.
            # 2. Formatting a task input for that agent (e.g. a StepRequestBody).
            # 3. Calling its execute_step or specific capability method.
            # 4. Handling the asynchronous response.

            self.orchestration_status_message = f"Task '{conceptual_first_task_name}' assigned to role '{first_agent_role}' (agent type: {first_agent_type_id}) in company '{first_company_name}'. Awaiting (simulated) completion."
            self.dv_mode = DVMode.ORCHESTRATING_TASKS # Transition to general task monitoring

            step.output = json.dumps({
                "status": "ORCHESTRATION_UNDERWAY",
                "message": self.orchestration_status_message,
                "current_target_company": first_company_name,
                "current_target_agent_role": first_agent_role,
                "current_target_agent_type": first_agent_type_id,
                "assigned_task_name": conceptual_first_task_name
            })
            step.is_last = True # Actual task execution by other agents takes time. DV waits for next trigger/update.
            # In a real system, DV might now periodically check status or wait for a callback.

        return step
