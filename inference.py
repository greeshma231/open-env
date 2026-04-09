#!/usr/bin/env python3
"""
CorpExpenseAudit inference agent using OpenAI-compatible API.

Matches OpenEnv STDOUT format:
  [START] task=<task> env=<env> model=<model>
  [STEP] step=<n> action=<action> reward=<r> done=<bool> error=<msg>
  [END] success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>

Supports:
- OpenAI API / Groq API / Hugging Face Router / Any OpenAI-compatible endpoint
- Local environment (direct instantiation) or remote API via HTTP

Environment variables:
- LLM Configuration:
  - API_BASE_URL: Base URL for LLM API (default: https://api.openai.com/v1)
  - MODEL_NAME: Model to use (default: gpt-4-turbo-preview)
  - OPENAI_API_KEY: OpenAI API key
  - GROQ_API_KEY: Groq API key
  - HF_TOKEN: Hugging Face token
- Environment Configuration:
  - ENVIRONMENT_BASE_URL: Base URL for CorpExpenseAudit API (default: http://localhost:7860)
    Set this to connect to Docker container running the API
    Leave unset to use local environment directly
"""

import os
import json
import sys
import time
from datetime import datetime
from typing import Optional, Any, Dict, List
import re

from dotenv import load_dotenv
import requests
from openai import OpenAI
from environment import CorpExpenseAudit
from graders import run_easy_grader, run_medium_grader, run_hard_grader, print_grader_results

# Load environment variables from .env file
load_dotenv()


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line to stdout."""
    error_val = error if error else "null"
    done_str = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


class ExpenseAuditAgent:
    """AI agent for expense claim auditing with OpenEnv format compliance."""
    
    def __init__(self, task_difficulty: str = "easy", max_steps: int = 50):
        """Initialize the agent."""
        self.task_difficulty = task_difficulty
        self.max_steps = max_steps

        self.HF_TOKEN = os.getenv("HF_TOKEN")
        
        # Get LLM API config
        api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        
        self.api_key = self._get_api_key() or "validation-only-key"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url if api_base_url else None
        )
        
        self.model = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        
        # 1. Check for remote config
        self.env_base_url = os.getenv("ENVIRONMENT_BASE_URL", "https://pooja52755-corps-expenseaudit-openenv.hf.space")
        self.use_remote_env = self.env_base_url and "http" in self.env_base_url
        
        

        # In your inference.py __init__
        try:
           self.env = CorpExpenseAudit(task_difficulty=task_difficulty)
        except Exception as e:
            print(f"[WARN] Local env init failed, will rely on remote: {e}")
        
        if self.use_remote_env:
             print(f"[INFO] Using remote environment API at {self.env_base_url}", file=sys.stderr)
        else:
             print(f"[INFO] Using local environment instance", file=sys.stderr)
        
        # Track claim processing state to fix "short-term memory" issue
        self.claim_states = {}  # claim_id -> {"inspected": bool, "categorized": bool, "decided": bool}
        
        # FIX: Error tracking to prevent infinite loops
        self.claim_errors = {}  # claim_id -> error count
        self.completed_claims = set()  # Claims that are fully processed
        self.blocked_claims = set()  # Claims that hit max errors
        self.last_error = None  # Last error from environment
        self.consecutive_errors = 0  # Track consecutive errors on same claim
        self.last_action = None  # Track last action for loop detection
        self.last_reward = None  # Track last reward for loop detection
        
        # AMNESIA FIX: Track full episode history for LLM context
        self.step_history = []  # List of {step, action_type, reward, error}
        
        self.step_count = 0
        self.rewards = []
    
    @staticmethod
    def _get_api_key() -> Optional[str]:
        """Get API key from environment variables."""
        # Try HF Token first (router)
        key = os.getenv("HF_TOKEN")
        if key:
            return key
        
        # Try Groq
        key = os.getenv("GROQ_API_KEY")
        if key:
            return key
        
        # Try OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        
        return None
    
    def run_audit(self) -> Dict[str, Any]:
        """Run the complete audit task with OpenEnv format compliance."""
        # Emit [START] line
        log_start(task=self.task_difficulty, env="CorpExpenseAudit", model=self.model)
        
        # Log model capabilities
        if "gpt-4o" in self.model.lower():
            pass
        elif "o1" in self.model.lower():
            pass
        else:
            pass
        
        # Reset environment (no seeding for truly random claim IDs)
        initial_state = self.env.reset()
        
        done = False
        success = False
        score = 0.0
        final_state = None
        
        try:
            # Main loop
            for step_num in range(1, self.max_steps + 1):
                if done:
                    break
                
                # Get agent's action
                action = self._get_agent_action(initial_state if step_num == 1 else None)
                
                if not action:
                    # Fallback to export if model fails repeatedly
                    action = {"action_type": "export_final_report", "action_data": {}}
                
                # Execute action in environment
                state, reward, done, info = self.env.step(action)
                final_state = state
                self.step_count = step_num
                self.rewards.append(reward)
                
                # STATE SYNC FIX: Update claim_state ONLY on success (no error)
                # This prevents loops where we keep trying the same action on same claim
                if not info.get("error"):
                    action_type = action.get("action_type")
                    claim_id = action.get("action_data", {}).get("claim_id")
                    
                    if claim_id and claim_id in self.claim_states:
                        if action_type == "inspect_claim":
                            self.claim_states[claim_id]["inspected"] = True
                            # CAPTURE TRUE AMOUNT AND DESCRIPTION from claim_details in info or state
                            details = info.get('claim_details') or state.get('claim_details')
                            if details:
                                self.claim_states[claim_id]['true_amount'] = float(details.get('amount', 100.0))
                                self.claim_states[claim_id]['description'] = details.get('description', '')
                            else:
                                # Fallback: try claims_summary
                                if 'claims_summary' in state:
                                    for claim in state['claims_summary']:
                                        if claim.get('claim_id') == claim_id:
                                            self.claim_states[claim_id]['true_amount'] = float(claim.get('amount', 100.0))
                                            self.claim_states[claim_id]['description'] = claim.get('description', '')
                                            break
                        elif action_type == "categorize_claim":
                            self.claim_states[claim_id]["categorized"] = True
                        elif action_type == "verify_gst":
                            self.claim_states[claim_id]["verified_gst"] = True
                            # CAPTURE GST STATUS from info
                            gst_status = info.get('gst_status')
                            if gst_status:
                                self.claim_states[claim_id]['gst_status'] = gst_status
                                # If non_compliant, mark for rejection
                                if gst_status == 'non_compliant':
                                    self.claim_states[claim_id]['should_reject'] = True
                        elif action_type in ["approve_claim", "reject_claim", "flag_fraud"]:
                            self.claim_states[claim_id]["decided"] = True
                            self.completed_claims.add(claim_id)  # Mark as completed so we don't repeat it
                
                # RATE LIMIT FIX: Add delay between requests (reduced to 0.7s for faster testing with smaller model)
                time.sleep(0.7)
                
                # AMNESIA FIX: Track this step in episode history for context
                self.step_history.append({
                    "step": step_num,
                    "action_type": action.get("action_type", "unknown"),
                    "reward": reward,
                    "error": info.get("error") if "error" in info else None
                })
                
                # Track last action and reward for loop detection
                self.last_action = action.get("action_type")
                self.last_reward = reward
                
                # FIX: Track errors to prevent infinite loops
                if "error" in info:
                    self.last_error = info["error"]
                    claim_id = action.get("action_data", {}).get("claim_id")
                    if claim_id:
                        self.claim_errors[claim_id] = self.claim_errors.get(claim_id, 0) + 1
                        # If claim has 3+ errors, mark it as blocked and force move to next
                        if self.claim_errors[claim_id] >= 3:
                            self.blocked_claims.add(claim_id)
                        self.consecutive_errors += 1
                    # Force break if too many consecutive errors (rate limit protection)
                    if self.consecutive_errors >= 5:
                        pass
                        break
                else:
                    # Success - reset error counter
                    self.consecutive_errors = 0
                    if action.get("action_type") == "export_final_report":
                        self.completed_claims.add("export_report")
                
                # Emit [STEP] line
                action_str = f"{action['action_type']}({action.get('action_data', {})})"
                error_msg = info.get("error") if "error" in info else None
                log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)
                
                # Check if audit is complete
                if done:
                    break
            
            # Generate final report if not done
            if not done and self.step_count >= self.max_steps:
                action = {"action_type": "export_final_report", "action_data": {}}
                state, reward, done, info = self.env.step(action)
                final_state = state
                self.step_count += 1
                self.rewards.append(reward)
                log_step(
                    step=self.step_count,
                    action=f"{action['action_type']}()",
                    reward=reward,
                    done=done,
                    error=info.get("error") if "error" in info else None
                )
            
            # Grade the task
            if self.task_difficulty == "easy":
                metrics = run_easy_grader(self.env)
            elif self.task_difficulty == "medium":
                metrics = run_medium_grader(self.env)
            else:
                metrics = run_hard_grader(self.env)
            
            score = metrics.final_score
            
            success = score >= 0.55  # Need 55%+ for true success (0.50 is borderline failure)
            
        except Exception as e:
            print(f"[ERROR] {str(e)}", file=sys.stderr)
            success = False
            score = 0.0
        
        finally:
            # Emit [END] line
            log_end(success=success, steps=self.step_count, score=score, rewards=self.rewards)
        
        return {
            "task_difficulty": self.task_difficulty,
            "steps_used": self.step_count,
            "final_score": score,
            "success": success,
            "total_reward": sum(self.rewards),
            "rewards": self.rewards
        }
    
    def _get_agent_action(self, initial_state: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get next action with ERROR FEEDBACK and AUTO-CLAIM-SWITCHING to fix loops."""
        import time
        
        state = initial_state or self.env.state_dict() if self.env.state else None
        
        if not state:
            return None
        
        pending = state['pending_claims']
        min_steps_required = int(state['max_steps'] * 0.6)
        
        if pending:
            pass
        
        # Force export only if BOTH conditions are met:
        # 1. No pending claims left, AND
        # 2. Enough steps have been used (60%+)
        if not pending:
            if state['current_step'] >= min_steps_required:
                pass
                return {
                    "action_type": "export_final_report",
                    "action_data": {},
                    "reasoning": "All claims processed and 60%+ steps used. Exporting final report."
                }
            else:
                # Pending is empty but not enough steps yet - this shouldn't happen in normal flow
                steps_remaining = state['max_steps'] - state['current_step']
                pass
                # Force export anyway since there's nothing left to do
                return {
                    "action_type": "export_final_report",
                    "action_data": {},
                    "reasoning": "All claims done, no more pending claims. Export required."
                }
        
        # Track completed claims for fraud detection (description + amount pairs)
        # This is used to detect duplicate/fraudulent claims
        self.completed_claim_signatures = set()  # Set of (description, amount) tuples for fraud detection
        for completed_id in self.completed_claims:
            if completed_id in self.claim_states:
                desc = self.claim_states[completed_id].get('description', '')
                amt = self.claim_states[completed_id].get('true_amount', 0)
                if desc and amt:
                    self.completed_claim_signatures.add((desc, float(amt)))
        
        # FIX #3: AUTO-SWITCH to next unblocked claim
        # Find first claim that isn't blocked, isn't completed, and isn't in self.completed_claims
        target_claim_id = None
        for claim_id in pending:
            if claim_id not in self.blocked_claims and claim_id not in self.completed_claims:
                target_claim_id = claim_id
                break
        
        # If all pending claims are blocked, export report (forced decision)
        if target_claim_id is None:
            return {
                "action_type": "export_final_report",
                "action_data": {},
                "reasoning": "All remaining claims are blocked or completed. Exporting report."
            }
        
        # Initialize tracking for this claim
        if target_claim_id not in self.claim_states:
            self.claim_states[target_claim_id] = {
                "inspected": False,
                "categorized": False,
                "verified_gst": False,
                "decided": False,
                "true_amount": None,  # Will be set when inspect_claim succeeds
                "description": "",  # Will be set when inspect_claim succeeds
                "gst_status": None,  # Will be set when verify_gst succeeds (compliant, non_compliant, etc.)
                "should_reject": False  # Will be True if GST is non_compliant
            }
        
        claim_state = self.claim_states[target_claim_id]
        
        # Determine what STAGE we're at for this claim
        if not claim_state["inspected"]:
            next_stage = "INSPECT"
        elif not claim_state["categorized"]:
            next_stage = "CATEGORIZE"
        elif not claim_state["verified_gst"]:
            next_stage = "VERIFY_GST"
        else:
            next_stage = "DECIDE"
        
        # AMNESIA FIX #1: Build complete episode history context
        history_context = ""
        if self.step_history:
            history_lines = []
            for entry in self.step_history[-5:]:  # Show last 5 steps (reduced from 10 to save tokens)
                step = entry['step']
                action = entry['action_type']
                reward = entry['reward']
                error = entry['error']
                error_str = f" | ERROR: {error}" if error else ""
                history_lines.append(f"  Step {step}: {action} → reward={reward:+.2f}{error_str}")
            
            history_context = f"""📋 EPISODE HISTORY (Last 5 steps):
{chr(10).join(history_lines)}

**LEARN FROM HISTORY**: Avoid actions that failed. Use errors to make better decisions.
If you see error 'stage violation', it means you skipped a required step. Go BACK to complete that stage first!
"""
        
        # Include last error with explicit prohibition
        error_context = ""
        if self.last_error:
            error_context = f"""
⚠️ LAST ACTION ERROR:
"{self.last_error}"

IF THE ERROR SAYS "already inspected":
  You ARE FORBIDDEN from inspecting that claim again!
  You MUST choose a different action:
  - Try: action_type="categorize_claim" (if not yet categorized)
  - Try: action_type="verify_gst" (if not yet verified)
  - Try: action_type="approve_claim" or "reject_claim" or "flag_fraud" (if ready to decide)
  - Or try a DIFFERENT claim_id entirely

IF THE ERROR SAYS "already categorized":
  You ARE FORBIDDEN from categorizing that claim again!
  Move to: action_type="verify_gst" or action_type="approve_claim", etc.
"""
        
        system_prompt = ("You are a Senior Fraud Auditor. You currently have " + str(len(pending)) + " claims left to audit.\n\n" +
                        "🚨🚨🚨 CRITICAL RULE - READ THIS CAREFULLY 🚨🚨🚨\n" +
                        "⛔️ DISQUALIFICATION WARNING ⛔️\n" +
                        "If you call 'export_final_report' while [PENDING_COUNT=" + str(len(pending)) + "] is GREATER THAN 0, you will BE DISQUALIFIED.\n" +
                        "This is not a suggestion. This is a HARD RULE.\n" +
                        "- Calling export early = INSTANT FAILURE\n" +
                        "- You will get success=false\n" +
                        "- Your audit will be REJECTED\n" +
                        "- No exceptions. No second chances.\n\n" +
                        "CURRENT PENDING CLAIMS TO PROCESS:\n" +
                        str(", ".join(pending[:10])) + ("..." if len(pending) > 10 else "") + "\n\n" +
                        "ACTION: You MUST process ALL " + str(len(pending)) + " claims before calling export.\n" +
                        "ONLY when pending_claims becomes 0 are you allowed to export.\n\n" +
                        "⚠️ ACCURACY PENALTIES FROM PREVIOUS RUNS:\n" +
                        "Step 36: Agent rejected a VALID claim without completing GST check → reward = -0.20\n" +
                        "Step 46: Agent picked WRONG category (not reading description carefully) → reward = -0.08\n" +
                        "Step 55: Agent's reasoning was incomplete (confused about categories) → reward = +0.08 (LOW, should be +0.15)\n" +
                        "LESSON: These mistakes cost 0.36 points total. Your score dropped from 0.89 to 0.53 because of hasty decisions.\n" +
                        "FIX: Slow down. Read descriptions word-by-word. Complete ALL 4 stages before deciding. DO NOT REJECT without GST verification.\n\n" +
                        "🎯 PRIORITY FIXES FOR COMMON MISTAKES:\n" +
                        "1. STATIONERY/PAPER/PENS/NOTEBOOKS\n" +
                        "   ❌ WRONG: Categorize as 'equipment'\n" +
                        "   ✅ RIGHT: Categorize as 'office_supplies'\n" +
                        "   Penalty for wrong: -0.08 per claim\n\n" +
                        "2. PERSONAL GROCERIES\n" +
                        "   ❌ WRONG: Categorize as 'meals' or 'miscellaneous'\n" +
                        "   ✅ RIGHT: Categorize as 'miscellaneous' and REJECT as not business expense\n" +
                        "   Penalty for wrong: -0.10 per claim\n\n" +
                        "3. WORKFLOW FOR EACH CLAIM:\n" +
                        "   Inspect → Categorize (read description carefully!) → Verify GST → Decide (Approve/Reject/Flag)\n\n" +
                        "REMEMBER: You have 80 steps. Use them wisely to audit EVERY claim WITHOUT EXCEPTION.\n" +
                        "Do not think about exporting until pending_claims = 0.\n" +
                        "You are currently at STAGE: " + str(next_stage) + "\n\n" +
                        str(history_context) + "\n\n" +
                        str(error_context) + "\n\n" +
                        "CRITICAL: Category accuracy matters! Correct = +0.15, Wrong = -0.08. That's 0.23 difference per claim!\n" +
                        "REWARD EXAMPLES:\n" +
                        "- Hotel booking -> 'travel' = +0.15\n" +
                        "- Flight reservation -> 'travel' = +0.15\n" +
                        "- Lunch at restaurant -> 'meals' = +0.15\n" +
                        "- Laptop purchase -> 'equipment' = +0.15\n" +
                        "- Accommodation fee -> 'accommodation' = +0.15\n" +
                        "- STATIONERY/PAPER/PENS -> 'office_supplies' = +0.15  ⭐ NOT equipment!\n" +
                        "- Notebooks, printer paper -> 'office_supplies' = +0.15  ⭐ NOT equipment!\n" +
                        "- Wrong lazy category = -0.08 penalty.\n" +
                        "⚠️ SPECIAL: Stationery categorized as 'equipment' = -0.08 PENALTY. Stationery MUST be 'office_supplies'!\n\n" +
                        "🎯 BIG FRAUD DETECTION REWARDS (These are WORTH THE EFFORT):\n" +
                        "- Correctly detect DUPLICATE claim (same amount + description) = +0.30 ⭐⭐⭐\n" +
                        "- Correctly detect FAKE GST invoice = +0.30 ⭐⭐⭐\n" +
                        "- Correctly flag other fraud patterns = +0.30 ⭐⭐⭐\n" +
                        "- These BIG FRAUD PATTERNS are usually at the END of the claim list!\n" +
                        "- If you export early, you miss these high-value rewards!\n" +
                        "- Example: Process 7/15 claims = lose +0.30 fraud detection reward = stuck at 0.50 score\n" +
                        "- Solution: Process ALL 15 claims = catch fraud = get +0.30 = score reaches 0.80+\n\n" +
                        "FOCUS: Start by reading the claim description carefully before picking a category.\n" +
                        "DO NOT QUIT EARLY. Big rewards are at the end!\n\n" +
                        "═══ MANDATORY WORKFLOW - YOU MUST FOLLOW THIS EXACT SEQUENCE ═══\n" +
                        "EVERY claim MUST complete these 4 stages IN THIS ORDER before moving to next claim:\n" +
                        "  Stage 1 → Stage 2 → Stage 3 → Stage 4 → ONLY THEN move to next claim\n" +
                        "Skipping stages will trigger 'stage violation' errors. Do NOT skip.\n\n" +
                        "1️⃣ INSPECT (Stage 1): Read claim details and REMEMBER THE AMOUNT\n" +
                        '   REQUIRED KEYS: action_type, action_data with claim_id\n' +
                        '   Example: {"action_type": "inspect_claim", "action_data": {"claim_id": "claim-123"}}\n' +
                        "   What you get: description, amount, date\n" +
                        "   CRITICAL: Write down the exact amount you see. You will need it in Stage 4.\n" +
                        "   AFTER: Move immediately to Stage 2 (CATEGORIZE). If you inspect again, FORBIDDEN = error\n\n" +
                        "2️⃣ CATEGORIZE (Stage 2): Read description carefully, assign category\n" +
                        "   Categories: travel, meals, accommodation, equipment, entertainment, miscellaneous, office_supplies\n" +
                        "   ⚠️ **READ THE DESCRIPTION WORD-BY-WORD** - most failures from lazy reading!\n" +
                        "   Examples of correct categorization:\n" +
                        "     - 'Flight to NYC' → 'travel'\n" +
                        "     - 'Hotel in Boston' → 'travel'\n" +
                        "     - 'Lunch at restaurant' → 'meals'\n" +
                        "     - 'Office stationery/pens/paper/notebooks' → 'office_supplies' ⭐ NOT 'equipment'!\n" +
                        "     - 'Laptop purchase' → 'equipment'\n" +
                        "     - 'Accommodation for visiting client' → 'accommodation'\n" +
                        "   ⚠️ **COMMON MISTAKE (Step 46)**: Picking 'equipment' for stationery = -0.08 penalty!\n" +
                        "   Penalties: Correct = +0.15, Wrong = -0.08. Difference = 0.23 per claim!\n" +
                        '   REQUIRED KEYS: action_type, action_data with claim_id, category, confidence\n' +
                        '   Example: {"action_type": "categorize_claim", "action_data": {"claim_id": "claim-123", "category": "travel", "confidence": 0.85}}\n' +
                        "   AFTER: Move immediately to Stage 3 (VERIFY_GST). If you categorize again, FORBIDDEN = error\n\n" +
                        "3️⃣ VERIFY_GST (Stage 3): Must complete before deciding to REJECT or APPROVE\n" +
                        '   REQUIRED KEYS: action_type, action_data with claim_id\n' +
                        '   Example: {"action_type": "verify_gst", "action_data": {"claim_id": "claim-123"}}\n' +
                        "   What you get: GST status (compliant, non_compliant, not_applicable, unverifiable)\n" +
                        "   ⚠️ **CRITICAL**: Do NOT reject or approve until you have verified GST!\n" +
                        "   ⚠️ **PAST MISTAKE (Step 36)**: Rejecting before GST verification = -0.20 penalty!\n" +
                        "   AFTER: Move immediately to Stage 4 (DECIDE). You are now ready to approve/reject/flag.\n\n" +
                        "4️⃣ DECIDE (Stage 4): Final decision based on all info from Stages 1-3\n" +
                        '   REQUIRED KEYS for approve: action_type, action_data with claim_id AND approved_amount\n' +
                        "   CRITICAL: approved_amount MUST match the amount you found in Stage 1 (INSPECT)\n" +
                        "   CRITICAL: If GST is non_compliant, you may want to flag_fraud instead of approve\n" +
                        '   Approve: {"action_type": "approve_claim", "action_data": {"claim_id": "claim-123", "approved_amount": 150.50}}\n' +
                        '   Reject: {"action_type": "reject_claim", "action_data": {"claim_id": "claim-123", "reason": "duplicate_claim"}}\n' +
                        '   Flag Fraud: {"action_type": "flag_fraud", "action_data": {"claim_id": "claim-123"}}\n' +
                        "   AFTER: Claim is DONE. Move to next claim_id. Start from Stage 1 again.\n\n" +
                        "⚠️ IF YOU SEE ERROR 'stage violation: tried [action], expected [stage]':\n" +
                        "  This means you skipped a required stage. Go BACK to the expected stage!\n" +
                        "  Example: error says 'expected [categorize_claim]' means you must categorize first\n" +
                        "  DO NOT SKIP STAGES!\n\n" +
                        "FORBIDDEN ACTIONS:\n" +
                        "🚫 DO NOT skip stages! If environment says 'stage violation', go back and complete the expected stage!\n" +
                        "🚫 DO NOT inspect the same claim twice. Once inspected, move to CATEGORIZE.\n" +
                        "🚫 DO NOT categorize the same claim twice. Once categorized, move to VERIFY_GST.\n" +
                        "🚫 DO NOT reject/approve/flag without completing VERIFY_GST first!\n" +
                        "  - Past mistake: Rejecting at Step 36 without GST check = -0.20 penalty\n" +
                        "🚫 DO NOT pick WRONG category. Read description carefully!\n" +
                        "  - Past mistake: Picking 'equipment' for stationery at Step 46 = -0.08 penalty\n" +
                        "  - Correct approach: Read 'stationery/pens/paper/notebooks' → IMMEDIATELY think 'office_supplies'\n" +
                        "🚫 STATIONERY/PAPER/PENS MUST be 'office_supplies', NOT 'equipment'. Wrong = -0.08!\n" +
                        "🚫 Do NOT default to 'miscellaneous' - think about what the claim is really for!\n" +
                        "🚫 DUPLICATE ENTRIES (same amount + same description) MUST be REJECTED with reason='duplicate_claim'\n" +
                        "🚫 DO NOT EXPORT until pending_claims list is EMPTY!\n" +
                        "🚫 NEVER omit the 'category' key in categorize_claim\n" +
                        "🚫 NEVER omit the 'approved_amount' key in approve_claim\n" +
                        "🚫 NEVER use uppercase categories like 'Travel' or 'TRAVEL', always use lowercase\n\n" +
                        "RULES:\n" +
                        '- Use LOWERCASE action names: "inspect_claim" not "INSPECT_CLAIM"\n' +
                        '- Use LOWERCASE categories: "travel" not "Travel"\n' +
                        "- Each claim needs inspect → categorize → verify → decide in order\n" +
                        "- Once decided, MOVE TO NEXT CLAIM - do not repeat\n" +
                        "- IF YOU GET AN ERROR, DON'T REPEAT IT - move to next stage\n" +
                        "- Inspecting/categorizing same claim twice = -0.05 penalty\n" +
                        "- WRONG CATEGORY = -0.08 penalty. RIGHT CATEGORY = +0.15 reward.\n\n" +
                        "RETURN FORMAT:\n" +
                        "Return ONLY valid JSON on one line. No markdown, no code blocks.\n" +
                        'GOOD: {"action_type": "inspect_claim", "action_data": {"claim_id": "claim-001"}}\n' +
                        'GOOD: {"action_type": "categorize_claim", "action_data": {"claim_id": "claim-001", "category": "travel", "confidence": 0.85}}\n' +
                        'GOOD: {"action_type": "verify_gst", "action_data": {"claim_id": "claim-001"}}\n' +
                        'GOOD: {"action_type": "approve_claim", "action_data": {"claim_id": "claim-001", "approved_amount": 250.75}}\n' +
                       'BAD: Picking miscellaneous without reading the description\n' +
                        'BAD: Missing category in categorize_claim\n' +
                        'BAD: Missing approved_amount in approve_claim\n' +
                        'BAD: Uppercase category like "Travel"\n' +
                        'BAD: Approving the same claim twice\n' +
                        'BAD: Markdown or code blocks\n' +
                        'BAD: Exporting before all claims are processed (AUDIT FAILURE!)\n\n' +
                        "⛔️ FINAL WARNING ABOUT EXPORT ⛔️\n" +
                        "export_final_report is ONLY allowed when BOTH:\n" +
                        "  1) pending_claims = 0 (all claims processed), AND\n" +
                        "  2) You've used 60%+ of steps (48+ out of 80)\n" +
                        "Currently: pending_claims = " + str(len(pending)) + " | Steps: " + str(state['current_step']) + "/" + str(state['max_steps']) + "\n" +
                        "If you try to export early, it will be REJECTED and you FAIL the audit.\n" +
                        "DO NOT EXPORT until:\n" +
                        "  ✓ All pending claims are processed, AND\n" +
                        "  ✓ You've used at least 48+ steps\n"
        )
        
        # FIX #2: Show completed and blocked claims so LLM knows to skip them
        claims_context = []
        for idx, claim_summary in enumerate(state['claims_summary'][:5]):
            cid = claim_summary['claim_id']
            
            # Show blocked status
            if cid in self.blocked_claims:
                status_str = "🚫 BLOCKED (too many errors, skip this)"
            else:
                cs = self.claim_states.get(cid, {})
                status_parts = []
                if cs.get("inspected"):
                    status_parts.append("✓Inspected")
                if cs.get("categorized"):
                    status_parts.append("✓Categorized")
                if cs.get("verified_gst"):
                    status_parts.append("✓GST-checked")
                if cs.get("decided"):
                    status_parts.append("✓DECIDED")
                
                status_str = " ".join(status_parts) if status_parts else "❌ NOT STARTED"
            
            line = f"- {cid}: {claim_summary['description'][:40]} | {status_str}"
            claims_context.append(line)
        
        claims_text = "\n".join(claims_context)
        
        # BUILD MEMORY CONTEXT: Include inspection results so LLM remembers what it found
        inspection_context = ""
        if next_stage in ["CATEGORIZE", "VERIFY_GST", "DECIDE"]:
            # LLM should remember what it discovered during INSPECT
            true_amount = claim_state.get('true_amount')
            description = claim_state.get('description')
            gst_status = claim_state.get('gst_status')
            
            memory_parts = []
            if true_amount:
                memory_parts.append(f"Amount: ${true_amount}")
            if description:
                memory_parts.append(f"Description: {description}")
            if gst_status and next_stage in ["VERIFY_GST", "DECIDE"]:
                memory_parts.append(f"GST Status: {gst_status}")
            
            if memory_parts:
                inspection_context = "📌 YOUR PREVIOUS INSPECTION RESULTS (DO NOT FORGET):\n" + "\n".join(memory_parts) + "\n\n"
        
        # Stage-specific prompts
        if next_stage == "INSPECT":
            action_instruction = f"Use action_type='inspect_claim' with claim_id='{target_claim_id}'. Look at ALL details."
        elif next_stage == "CATEGORIZE":
            action_instruction = f"Use action_type='categorize_claim' with claim_id='{target_claim_id}'. Pick ONE category: travel, meals, accommodation, equipment, entertainment, or miscellaneous."
        elif next_stage == "VERIFY_GST":
            action_instruction = f"Use action_type='verify_gst' with claim_id='{target_claim_id}'. Status must be: compliant, non_compliant, not_applicable, or unverifiable."
        else:  # DECIDE
            # LLM should see GST status in inspection_context and decide for itself
            action_instruction = f"Use ONE of: action_type='approve_claim' OR 'reject_claim' OR 'flag_fraud' with claim_id='{target_claim_id}'. Remember the inspection results above (amount, GST status, etc) and make your decision."

        # Build pending claims list for display
        pending_claims_display = ", ".join([f"{p}" for p in pending[:5]])
        if len(pending) > 5:
            pending_claims_display += f" ... and {len(pending) - 5} more"
        
        # Calculate step percentages
        step_percent = int((state['current_step'] / state['max_steps']) * 100)
        steps_remaining = state['max_steps'] - state['current_step']
        min_steps_required = int(state['max_steps'] * 0.6)
        
        # CRITICAL: Build export prevention message
        export_prevention = ""
        has_pending = len(pending) > 0
        insufficient_steps = state['current_step'] < min_steps_required
        
        if has_pending or insufficient_steps:
            export_prevention = "\n🔴 **EXPORT IS BLOCKED** 🔴\n"
            if has_pending:
                export_prevention += f"❌ You still have {len(pending)} PENDING CLAIMS: {pending_claims_display}\n"
            if insufficient_steps:
                export_prevention += f"❌ Only used {state['current_step']}/{state['max_steps']} steps ({step_percent}%). Need 60%+ ({min_steps_required}+).\n"
            export_prevention += "\n⛔️ YOU ARE STRICTLY FORBIDDEN FROM EXPORTING!\n"
            export_prevention += f"✅ KEEP WORKING! You have {steps_remaining} steps remaining.\n"
        
        user_message = (f"Step {state['current_step']}/{state['max_steps']} ({step_percent}% complete) | {steps_remaining} steps remaining\n\n" +
                       str(export_prevention) + "\n" +
                       "STAGE: " + str(next_stage) + "\n" +
                       "CURRENT TARGET: " + str(target_claim_id) + "\n\n" +
                       str(action_instruction) + "\n\n" +
                       str(inspection_context) +
                       "Pending claims: " + str(len(pending)) + "\n" +
                       "Processed: " + str(len(state['claims_summary']) - len(pending)) + "/" + str(len(state['claims_summary'])) + "\n\n" +
                       "Claims Status:\n" +
                       claims_text + "\n\n" +
                       "NOTE: If current target is blocked (🚫), the system will automatically switch to next claim next step.\n\n" +
                       "════════════════════════════════════════════\n" +
                       "JSON FORMAT EXAMPLES FOR THIS STAGE (COPY EXACTLY):\n" +
                       "════════════════════════════════════════════\n\n" +
                       "IF STAGE = INSPECT:\n" +
                       '  {"action_type": "inspect_claim", "action_data": {"claim_id": "' + target_claim_id + '"}}\n' +
                       "  (Remember the amount you see - you'll need it for DECIDE stage!)\n\n" +
                       "IF STAGE = CATEGORIZE (MUST HAVE: claim_id, category, confidence):\n" +
                       '  {"action_type": "categorize_claim", "action_data": {"claim_id": "' + target_claim_id + '", "category": "travel", "confidence": 0.85}}\n' +
                       "  Allowed categories: travel, meals, accommodation, equipment, entertainment, miscellaneous\n" +
                       "  IMPORTANT: Pick the BEST match based on the claim, not 'miscellaneous'!\n" +
                       "  IMPORTANT: category is REQUIRED - do NOT forget it!\n\n" +
                       "IF STAGE = VERIFY_GST:\n" +
                       '  {"action_type": "verify_gst", "action_data": {"claim_id": "' + target_claim_id + '"}}\n\n' +
                       "IF STAGE = DECIDE:\n" +
                       '  {"action_type": "approve_claim", "action_data": {"claim_id": "' + target_claim_id + '", "approved_amount": 125.50}}\n' +
                       "  ^^^ CRITICAL: Include approved_amount from the inspect result! ^^^\n" +
                       "  OR\n" +
                       '  {"action_type": "reject_claim", "action_data": {"claim_id": "' + target_claim_id + '"}}\n' +
                       "  OR\n" +
                       '  {"action_type": "flag_fraud", "action_data": {"claim_id": "' + target_claim_id + '"}}\n\n' +
                       "════════════════════════════════════════════\n" +
                       "CRITICAL RULES:\n" +
                       "1. Return ONLY the JSON object on ONE line\n" +
                       "2. No markdown, no code blocks, no explanation\n" +
                       "3. For CATEGORIZE: Always include the 'category' key - pick the BEST category, not miscellaneous\n" +
                       "4. For APPROVE: Always include 'approved_amount' from the inspect step\n" +
                       "5. Use lowercase everywhere: 'travel' not 'Travel'\n" +
                       "6. If you get an error, move to the NEXT stage (don't repeat)\n" +
                       "════════════════════════════════════════════"
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # FIX: Add small delay to prevent rate limiting
            time.sleep(0.1)
            
            # Optimize parameters based on model capabilities
            # Note: Extended thinking requires model-specific API support and SDK version
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,  # Lower temperature for more consistent reasoning
                "max_tokens": 500,    # Sufficient for JSON action responses
                "top_p": 0.9,        # Reduce randomness for better accuracy
            }
            
            # For reasoning-focused models, use better parameters
            if "gpt-4o" in self.model.lower():
                api_kwargs["temperature"] = 0.5  # Slightly higher for creativity in categorization
                pass
            elif "o1" in self.model.lower():
                # o1 models don't support temperature parameter
                api_kwargs.pop("temperature")
                api_kwargs.pop("top_p")
            else:
                pass
            
            response = self.client.chat.completions.create(**api_kwargs)
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response - GREEDY to match nested braces correctly
            # BUG FIX: Changed r'\{[\s\S]*?\}' (non-greedy) to r'\{[\s\S]*\}' (greedy)
            # Non-greedy stops at first }, missing the final } of outer JSON object
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                action_json = json_match.group()
                try:
                    action = json.loads(action_json)
                except json.JSONDecodeError as e:
                    return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                
                # ENFORCE LOWERCASE action_type
                if "action_type" in action:
                    action["action_type"] = action["action_type"].lower()
                
                # === CRITICAL: Code-level enforcement ===
                # 1. Block premature export if there are pending claims OR not enough steps used
                if action.get("action_type") == "export_final_report":
                    min_steps_required = int(state['max_steps'] * 0.6)  # Must use 60% of available steps
                    steps_remaining = state['max_steps'] - state['current_step']
                    
                    # Check 1: Pending claims still remain
                    if len(pending) > 0:
                        return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                    
                    # Check 2: Not enough steps used yet
                    if state['current_step'] < min_steps_required:
                        return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                
                # 2. Block stage skipping - force correct action
                expected_actions = {
                    "INSPECT": "inspect_claim",
                    "CATEGORIZE": "categorize_claim",
                    "VERIFY_GST": "verify_gst",
                    "DECIDE": ["approve_claim", "reject_claim", "flag_fraud"]
                }
                
                expected = expected_actions.get(next_stage, [])
                if isinstance(expected, str):
                    expected = [expected]
                
                if action.get("action_type") not in expected and action.get("action_type") != "export_final_report":
                    # LLM tried to skip stages! Override it
                    return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                
                # 3. Prevent repeating same action on same claim
                if self.last_action == action.get("action_type") and self.last_action and self.last_reward < 0:
                    # Same action as last time AND last reward was negative = infinite loop!
                    return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                
                # Ensure action_data exists and has claim_id
                if "action_data" not in action:
                    action["action_data"] = {}
                
                if action.get("action_type") != "export_final_report":
                    action["action_data"]["claim_id"] = target_claim_id
                
                # VALIDATION ONLY (no forcing):
                # 1. Check categorize_claim has required fields
                if action.get("action_type") == "categorize_claim":
                    if "category" not in action.get("action_data", {}):
                        return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                    # Accept LLM's category as-is (even if wrong - it learns from negative reward)
                
                # 2. Check approve_claim has required fields
                if action.get("action_type") == "approve_claim":
                    if "approved_amount" not in action.get("action_data", {}):
                        return self._fallback_action(state, next_stage, target_claim_id, claim_state)
                    # Accept LLM's amount as-is (even if hallucinated - it learns from negative reward)
                
                # FINAL VERIFICATION: Log the complete action before returning
                
                # IMPORTANT: Do NOT update claim_state here! Update only after successful env.step() in run_audit
                # This prevents premature state updates that cause re-inspection loops
                
                return action
            else:
                return self._fallback_action(state, next_stage, target_claim_id, claim_state)
            
        except Exception as e:
            return self._fallback_action(state, next_stage, target_claim_id, claim_state)
    
    def _fallback_action(self, state: Dict[str, Any], stage: str, claim_id: str, claim_state: Dict) -> Dict:
        """Smart fallback that uses stored memory for accurate categorization and amounts."""
        if stage == "INSPECT":
            return {
                "action_type": "inspect_claim",
                "action_data": {"claim_id": claim_id},
                "reasoning": "Fallback: Start by inspecting"
            }
        elif stage == "CATEGORIZE":
            # Smart categorization using STORED description from memory (not from state)
            category = "miscellaneous"  # Default to miscellaneous
            
            # Get description from memory instead of searching state
            description = self.claim_states[claim_id].get('description', '').lower()
            
            # PRIORITY 1: Personal/Non-business items (should be categorized as miscellaneous)
            if any(kw in description for kw in ['personal', 'grocery', 'groceries', 'household', 'private']):
                category = "miscellaneous"
            # PRIORITY 2: STATIONERY/PAPER/OFFICE SUPPLIES (NOT equipment!)
            elif any(kw in description for kw in ['stationery', 'paper', 'pen', 'notebook', 'printer paper', 'pens', 'stationary', 'supplies', 'office supply']):
                category = "office_supplies"
            # PRIORITY 3: Travel keywords (highest priority for most corporate expenses)
            elif any(kw in description for kw in ['cab', 'fare', 'flight', 'hotel', 'train', 'uber', 'taxi', 'stay', 'booking', 'travel', 'airline', 'airfare']):
                category = "travel"
            # PRIORITY 4: Meals keywords
            elif any(kw in description for kw in ['meal', 'food', 'lunch', 'dinner', 'breakfast', 'restaurant', 'cafe', 'coffee', 'dining', 'business meal']):
                category = "meals"
            # PRIORITY 5: Equipment keywords (NOT stationery!)
            elif any(kw in description for kw in ['laptop', 'monitor', 'keyboard', 'software', 'mouse', 'computer', 'tablet', 'phone', 'hardware', 'equipment']):
                category = "equipment"
            # PRIORITY 6: Accommodation keywords
            elif any(kw in description for kw in ['accommodation', 'lodging', 'residence', 'apartment', 'room', 'hostel', 'guest']):
                category = "accommodation"
            # PRIORITY 7: Entertainment keywords
            elif any(kw in description for kw in ['entertainment', 'movie', 'concert', 'event', 'show', 'ticket', 'theater']):
                category = "entertainment"
            
            return {
                "action_type": "categorize_claim",
                "action_data": {
                    "claim_id": claim_id,
                    "category": category,
                    "confidence": 0.85
                },
                "reasoning": f"Fallback: Smart categorization to '{category}' based on stored keywords"
            }
        elif stage == "VERIFY_GST":
            return {
                "action_type": "verify_gst",
                "action_data": {"claim_id": claim_id},
                "reasoning": "Fallback: Verify GST status"
            }
        else:  # DECIDE
            # Check for fraud (duplicate description + amount)
            current_desc = self.claim_states[claim_id].get('description', '')
            current_amt = self.claim_states[claim_id].get('true_amount', 0)
            
            # Build fraud detection set
            if not hasattr(self, 'completed_claim_signatures'):
                self.completed_claim_signatures = set()
                for cid in self.completed_claims:
                    if cid in self.claim_states:
                        d = self.claim_states[cid].get('description', '')
                        a = self.claim_states[cid].get('true_amount', 0)
                        if d and a:
                            self.completed_claim_signatures.add((d, float(a)))
            
            # Check if this claim is a duplicate
            if current_desc and current_amt:
                claim_sig = (current_desc, float(current_amt))
                if claim_sig in self.completed_claim_signatures:
                    return {
                        "action_type": "flag_fraud",
                        "action_data": {"claim_id": claim_id},
                        "reasoning": "Fallback: Flagging duplicate claim as fraud"
                    }
            
            # Check for personal/non-business items - REJECT them
            current_desc = self.claim_states[claim_id].get('description', '').lower()
            if any(kw in current_desc for kw in ['personal', 'grocery', 'groceries', 'household', 'private']):
                return {
                    "action_type": "reject_claim",
                    "action_data": {"claim_id": claim_id, "reason": "non_business_expense"},
                    "reasoning": "Fallback: Reject personal/non-business expense"
                }
            
            # Check GST status
            gst_status = self.claim_states[claim_id].get('gst_status')
            if gst_status == 'non_compliant':
                return {
                    "action_type": "reject_claim",
                    "action_data": {"claim_id": claim_id, "reason": "non_compliant_gst"},
                    "reasoning": "Fallback: Reject due to non-compliant GST"
                }
            
            # Use STORED true_amount instead of hardcoded 100.0
            true_amount = self.claim_states[claim_id].get('true_amount', 100.0)
            return {
                "action_type": "approve_claim",
                "action_data": {"claim_id": claim_id, "approved_amount": float(true_amount)},
                "reasoning": f"Fallback: Approve with stored amount {true_amount}"
            }


def main():
    """Main entry point - run audits and emit OpenEnv format."""
    difficulties = ["easy", "medium", "hard"]
    #difficulties = ["hard"]
    results = []
    
    for difficulty in difficulties:
        try:
            # Use appropriate step limits based on difficulty
            if difficulty == "easy":
                max_steps = 40
            elif difficulty == "medium":
                max_steps = 80
            else:  # hard
                max_steps = 120
            
            agent = ExpenseAuditAgent(task_difficulty=difficulty, max_steps=max_steps)
            result = agent.run_audit()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to run {difficulty} task: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Exit with appropriate code
    success_count = sum(1 for r in results if r.get("success", False))
    exit_code = 0 if success_count >= 2 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
