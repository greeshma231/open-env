"""CorpExpenseAudit OpenEnv environment implementation."""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import random
import numpy as np
from models import (
    ExpenseClaim, AuditState, ClaimCategory, ClaimStatus, GSTStatus,
    FraudType, ActionInspectClaim, ActionCategorizeClaim, ActionVerifyGST,
    ActionFlagFraud, ActionApproveClaim, ActionRejectClaim, ActionRequestMoreInfo,
    ActionExportReport
)


class CorpExpenseAudit:
    """Main environment for expense claim auditing."""
    
    def __init__(self, task_difficulty: str = "easy"):
        """
        Initialize the environment.
        
        Args:
            task_difficulty: "easy", "medium", or "hard"
        """
        self.task_difficulty = task_difficulty
        self.state: Optional[AuditState] = None
        self.seed_value = None
        
    def seed(self, seed: int = None):
        """Set random seed for reproducibility."""
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial state."""

        # Only seed if explicitly set
        if self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)
        
        # Generate claims based on difficulty
        if self.task_difficulty == "easy":
            claims = self._generate_easy_claims()
            max_steps = 40
        elif self.task_difficulty == "medium":
            claims = self._generate_medium_claims()
            max_steps = 80
        elif self.task_difficulty == "hard":
            claims = self._generate_hard_claims()
            max_steps = 120
        else:
            raise ValueError(f"Unknown difficulty: {self.task_difficulty}")
        
        self.state = AuditState(
            task_id=f"task_{self.task_difficulty}_{int(datetime.now().timestamp())}",
            task_difficulty=self.task_difficulty,
            all_claims=claims,
            pending_claims=[c.claim_id for c in claims],
            reviewed_decisions={},
            current_step=0,
            max_steps=max_steps,
            total_reward=0.0
        )
        
        return self.state_dict()

    def state(self) -> Dict[str, Any]:
        """Return current state as dictionary."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        return {
            "task_id": self.state.task_id,
            "task_difficulty": self.state.task_difficulty,
            "current_step": self.state.current_step,
            "max_steps": self.state.max_steps,
            "pending_claims": self.state.pending_claims,
            "reviewed_count": len(self.state.reviewed_decisions),
            "total_claims": len(self.state.all_claims),
            "claims_summary": [
                {
                    "claim_id": c.claim_id,
                    "employee_id": c.employee_id,
                    "amount": c.amount,
                    "claimed_category": c.claimed_category,
                    "description": c.description,
                    "merchant_name": c.merchant_name,
                    "is_fraud": c.is_fraud,
                    "date_of_expense": c.date_of_expense.isoformat(),
                    "has_gst_invoice": c.has_gst_invoice
                }
                for c in self.state.all_claims
            ],
            "total_reward": self.state.total_reward,
            "audit_complete": self.state.audit_complete,
            "final_accuracy": self.state.final_accuracy
        }

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: Dictionary containing action_type and action_data
            
        Returns:
            (state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.state.current_step += 1
        reward = 0.0
        done = False
        info = {"action": action, "step": self.state.current_step}
        
        action_type = action.get("action_type", "")
        action_data = action.get("action_data", {})
        
        try:
            if action_type == "inspect_claim":
                reward, info = self._handle_inspect_claim(action_data, info)
            elif action_type == "categorize_claim":
                reward, info = self._handle_categorize_claim(action_data, info)
            elif action_type == "verify_gst":
                reward, info = self._handle_verify_gst(action_data, info)
            elif action_type == "flag_fraud":
                reward, info = self._handle_flag_fraud(action_data, info)
            elif action_type == "approve_claim":
                reward, info = self._handle_approve_claim(action_data, info)
            elif action_type == "reject_claim":
                reward, info = self._handle_reject_claim(action_data, info)
            elif action_type == "request_more_info":
                reward, info = self._handle_request_info(action_data, info)
            elif action_type == "export_final_report":
                reward, info, done = self._handle_export_report(action_data, info)
            else:
                reward = -0.05
                info["error"] = f"Unknown action_type: {action_type}"
        except Exception as e:
            reward = -0.10
            info["error"] = str(e)
        
        # Step penalty for inefficiency
        if not done and self.state.current_step > 10 and self.state.current_step % 5 == 0:
            reward -= 0.02
        
        # Check if max steps exceeded
        if self.state.current_step >= self.state.max_steps:
            done = True
            info["reason"] = "max_steps_exceeded"
            reward -= 0.15
        
        self.state.total_reward += reward
        self.state.step_rewards.append(reward)
        info["total_reward"] = self.state.total_reward
        
        return self.state_dict(), reward, done, info

    def _get_claim_by_id(self, claim_id: str) -> Optional[ExpenseClaim]:
        """Retrieve a claim by ID."""
        for claim in self.state.all_claims:
            if claim.claim_id == claim_id:
                return claim
        return None

    def _handle_inspect_claim(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle inspect_claim action."""
        claim_id = action_data.get("claim_id")
        if not claim_id:
            return -0.05, {**info, "error": "claim_id required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        # REPETITION PENALTY: Prevent inspecting same claim multiple times
        inspection_count = self.state.inspections.get(claim_id, 0)
        if inspection_count > 0:
            # Already inspected this claim! Move to categorization instead
            return -0.05, {**info, "error": f"Claim {claim_id} already inspected {inspection_count} time(s). Move to categorize/verify/decide."}
        
        # First inspection - small reward for investigation
        self.state.inspections[claim_id] = inspection_count + 1
        reward = 0.02
        
        info["claim_details"] = {
            "claim_id": claim.claim_id,
            "employee_id": claim.employee_id,
            "amount": claim.amount,
            "claimed_category": claim.claimed_category,
            "correct_category": claim.correct_category,
            "description": claim.description,
            "merchant_name": claim.merchant_name,
            "merchant_city": claim.merchant_city,
            "date_of_expense": claim.date_of_expense.isoformat(),
            "has_gst_invoice": claim.has_gst_invoice,
            "gst_invoice_valid": claim.gst_invoice_valid,
            "policy_compliant": claim.policy_compliant,
            "is_fraud": claim.is_fraud,
            "fraud_types": [f.value for f in claim.fraud_types],
            "mileage_claimed": claim.mileage_claimed
        }
        
        return reward, info

    def _handle_categorize_claim(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle categorize_claim action."""
        claim_id = action_data.get("claim_id")
        category = action_data.get("category")
        confidence = action_data.get("confidence", 0.5)
        
        if not claim_id or not category:
            return -0.05, {**info, "error": "claim_id and category required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        # PENALTY: Prevent reward farming by categorizing the same claim twice
        if claim_id in self.state.categorizations:
            # Already categorized this claim! Move on to approval/rejection decision
            return -0.05, {**info, "error": f"Claim {claim_id} already categorized. Move to approve/reject decision."}
        
        # Check if categorization is correct
        is_correct = category.lower() == claim.correct_category.lower()
        reward = 0.15 * confidence if is_correct else -0.08
        
        self.state.categorizations[claim_id] = category
        info["categorization_correct"] = is_correct
        info["confidence_weighted"] = confidence
        
        return reward, info

    def _handle_verify_gst(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle verify_gst action."""
        claim_id = action_data.get("claim_id")
        
        if not claim_id:
            return -0.05, {**info, "error": "claim_id required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        # Determine GST status
        if not claim.has_gst_invoice:
            status = GSTStatus.NOT_APPLICABLE.value
            reward = 0.10
        elif claim.gst_invoice_valid:
            status = GSTStatus.COMPLIANT.value
            reward = 0.20
        else:
            status = GSTStatus.NON_COMPLIANT.value
            reward = 0.15  # Reward for correct detection
        
        self.state.gst_verifications[claim_id] = status
        info["gst_status"] = status
        
        return reward, info

    def _handle_flag_fraud(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle flag_fraud action."""
        claim_id = action_data.get("claim_id")
        reason = action_data.get("reason", "")
        fraud_types = action_data.get("fraud_types", [])
        
        if not claim_id:
            return -0.05, {**info, "error": "claim_id required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        self.state.fraud_flags[claim_id] = reason
        
        # Remove from pending since this claim is now decided
        if claim_id in self.state.pending_claims:
            self.state.pending_claims.remove(claim_id)
        
        # Correct fraud detection
        if claim.is_fraud:
            reward = 0.30
            info["fraud_correctly_detected"] = True
        else:
            reward = -0.25  # False positive
            info["fraud_correctly_detected"] = False
        
        return reward, info

    def _handle_approve_claim(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle approve_claim action."""
        claim_id = action_data.get("claim_id")
        approved_amount = action_data.get("approved_amount")
        
        if not claim_id or approved_amount is None:
            return -0.05, {**info, "error": "claim_id and approved_amount required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        self.state.approvals[claim_id] = approved_amount
        
        # Remove from pending since this claim is now decided
        if claim_id in self.state.pending_claims:
            self.state.pending_claims.remove(claim_id)
        
        # Heavy penalty for approving fraud
        if claim.is_fraud:
            reward = -0.40
            info["approval_was_fraud"] = True
        else:
            # Reward based on accuracy of amount
            amount_diff = abs(approved_amount - claim.amount)
            amount_accuracy = max(0, 1 - (amount_diff / claim.amount)) if claim.amount > 0 else 1.0
            reward = 0.25 * amount_accuracy
            info["approval_was_fraud"] = False
            info["amount_accuracy"] = amount_accuracy
        
        return reward, info

    def _handle_reject_claim(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle reject_claim action."""
        claim_id = action_data.get("claim_id")
        reason = action_data.get("reason", "")
        
        if not claim_id:
            return -0.05, {**info, "error": "claim_id required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        self.state.rejections[claim_id] = reason
        
        # Remove from pending since this claim is now decided
        if claim_id in self.state.pending_claims:
            self.state.pending_claims.remove(claim_id)
        
        # Correct rejection of fraudulent claim
        if claim.is_fraud:
            reward = 0.30
            info["rejection_was_correct_fraud"] = True
        elif not claim.policy_compliant:
            reward = 0.20
            info["rejection_was_policy_violation"] = True
        else:
            # Penalty for rejecting valid claim
            reward = -0.20
            info["rejection_was_valid_claim"] = True
        
        return reward, info

    def _handle_request_info(self, action_data: Dict, info: Dict) -> tuple[float, Dict]:
        """Handle request_more_info action."""
        claim_id = action_data.get("claim_id")
        information_needed = action_data.get("information_needed", "")
        
        if not claim_id:
            return -0.05, {**info, "error": "claim_id required"}
        
        claim = self._get_claim_by_id(claim_id)
        if not claim:
            return -0.05, {**info, "error": f"claim_id {claim_id} not found"}
        
        self.state.info_requests[claim_id] = information_needed
        
        # Small reward for appropriate information requests
        reward = 0.05
        info["info_request_noted"] = True
        
        return reward, info

    def _handle_export_report(self, action_data: Dict, info: Dict) -> tuple[float, Dict, bool]:
        """Handle export_final_report action."""
        # Calculate final metrics
        total_claims = len(self.state.all_claims)
        correct_categorizations = 0
        correctly_detected_fraud = 0
        correctly_rejected_fraudulent = 0
        incorrectly_approved_fraudulent = 0
        correctly_approved_valid = 0
        gst_correct = 0
        
        for claim in self.state.all_claims:
            # Categorization check
            if claim.claim_id in self.state.categorizations:
                if self.state.categorizations[claim.claim_id].lower() == claim.correct_category.lower():
                    correct_categorizations += 1
            
            # Fraud detection check
            if claim.claim_id in self.state.fraud_flags:
                if claim.is_fraud:
                    correctly_detected_fraud += 1
                    if claim.claim_id in self.state.rejections:
                        correctly_rejected_fraudulent += 1
            
            if claim.claim_id in self.state.approvals and claim.is_fraud:
                incorrectly_approved_fraudulent += 1
            
            if claim.claim_id in self.state.approvals and not claim.is_fraud:
                correctly_approved_valid += 1
            
            # GST check
            if claim.claim_id in self.state.gst_verifications:
                status = self.state.gst_verifications[claim.claim_id]
                if claim.has_gst_invoice and status == GSTStatus.COMPLIANT.value and claim.gst_invoice_valid:
                    gst_correct += 1
                elif not claim.has_gst_invoice and status == GSTStatus.NOT_APPLICABLE.value:
                    gst_correct += 1
        
        # Calculate accuracy score
        categorization_accuracy = correct_categorizations / total_claims if total_claims > 0 else 0
        fraud_detection_rate = correctly_detected_fraud / sum(1 for c in self.state.all_claims if c.is_fraud) if any(c.is_fraud for c in self.state.all_claims) else 1.0
        gst_accuracy = gst_correct / total_claims if total_claims > 0 else 0
        
        # Penalty for approving fraudulent claims (softened to allow recovery from missed frauds)
        fraud_approval_penalty = incorrectly_approved_fraudulent * 0.10
        
        # Final accuracy calculation
        final_accuracy = (
            0.3 * categorization_accuracy +
            0.4 * fraud_detection_rate +
            0.3 * gst_accuracy -
            fraud_approval_penalty
        )
        final_accuracy = max(0.0, min(1.0, final_accuracy))
        
        # Reward based on accuracy
        reward = final_accuracy * 0.5 - 0.05  # Base reward with small penalty for report generation
        
        self.state.final_accuracy = final_accuracy
        self.state.audit_complete = True
        self.state.final_report = {
            "total_claims_processed": total_claims,
            "correct_categorizations": correct_categorizations,
            "correctly_detected_fraud": correctly_detected_fraud,
            "incorrectly_approved_fraudulent": incorrectly_approved_fraudulent,
            "correctly_approved_valid": correctly_approved_valid,
            "gst_accuracy": gst_accuracy,
            "categorization_accuracy": categorization_accuracy,
            "fraud_detection_rate": fraud_detection_rate,
            "final_accuracy": final_accuracy
        }
        
        info["report_generated"] = True
        info["final_metrics"] = self.state.final_report
        
        return reward, info, True

    def _generate_easy_claims(self) -> List[ExpenseClaim]:
        """Generate 8-10 simple claims for easy task."""
        claims = []
        base_date = datetime.now() - timedelta(days=7)
        
        # Simple travel claims
        for i in range(5):
            claims.append(ExpenseClaim(
                employee_id=f"EMP{100+i:03d}",
                amount=1500 + i*200,
                claimed_category="travel",
                correct_category="travel",
                description=f"Cab fare to office - Day {i+1}",
                date_submitted=base_date + timedelta(days=i),
                date_of_expense=base_date + timedelta(days=i),
                has_gst_invoice=True,
                gst_invoice_valid=True,
                merchant_name=f"Taxi Company {i+1}",
                merchant_city="Mumbai",
                policy_compliant=True,
                is_fraud=False
            ))
        
        # Simple meals
        for i in range(3):
            claims.append(ExpenseClaim(
                employee_id=f"EMP{200+i:03d}",
                amount=500 + i*100,
                claimed_category="meals",
                correct_category="meals",
                description=f"Business lunch - Client meeting",
                date_submitted=base_date + timedelta(days=i+1),
                date_of_expense=base_date + timedelta(days=i+1),
                has_gst_invoice=True,
                gst_invoice_valid=True,
                merchant_name=f"Restaurant {i+1}",
                merchant_city="Mumbai",
                policy_compliant=True,
                is_fraud=False
            ))
        
        # One office supplies - valid
        claims.append(ExpenseClaim(
            employee_id="EMP300",
            amount=2000,
            claimed_category="office_supplies",
            correct_category="office_supplies",
            description="Stationery and printer paper",
            date_submitted=base_date + timedelta(days=5),
            date_of_expense=base_date + timedelta(days=5),
            has_gst_invoice=True,
            gst_invoice_valid=True,
            merchant_name="Office Depot",
            merchant_city="Mumbai",
            policy_compliant=True,
            is_fraud=False
        ))
        
        return claims

    def _generate_medium_claims(self) -> List[ExpenseClaim]:
        """Generate 12-15 mixed claims with missing receipts and GST validation."""
        claims = self._generate_easy_claims()
        base_date = datetime.now() - timedelta(days=14)
        
        # Add more complex claims
        # Missing GST invoice
        claims.append(ExpenseClaim(
            employee_id="EMP401",
            amount=1200,
            claimed_category="travel",
            correct_category="travel",
            description="Flight ticket - Delhi business trip",
            date_submitted=base_date + timedelta(days=7),
            date_of_expense=base_date + timedelta(days=6),
            has_gst_invoice=False,
            gst_invoice_valid=False,
            merchant_name="Air India",
            merchant_city="New Delhi",
            policy_compliant=True,
            is_fraud=False,
            metadata={"ticket_number": "AI123456"}
        ))
        
        # Mismatched category
        claims.append(ExpenseClaim(
            employee_id="EMP402",
            amount=3000,
            claimed_category="entertainment",
            correct_category="meals",
            description="Team dinner event",
            date_submitted=base_date + timedelta(days=8),
            date_of_expense=base_date + timedelta(days=8),
            has_gst_invoice=True,
            gst_invoice_valid=True,
            merchant_name="5-Star Hotel",
            merchant_city="Mumbai",
            policy_compliant=True,
            is_fraud=False
        ))
        
        # Invalid GST invoice
        claims.append(ExpenseClaim(
            employee_id="EMP403",
            amount=2500,
            claimed_category="office_supplies",
            correct_category="office_supplies",
            description="Equipment purchase",
            date_submitted=base_date + timedelta(days=9),
            date_of_expense=base_date + timedelta(days=9),
            has_gst_invoice=True,
            gst_invoice_valid=False,
            merchant_name="Unknown Supplier",
            merchant_city="Bangalore",
            policy_compliant=True,
            is_fraud=True,
            fraud_types=[FraudType.FAKE_GST_INVOICE],
            fraudulent_reason="Fake GST invoice detected"
        ))
        
        # Personal vs business - policy violation
        claims.append(ExpenseClaim(
            employee_id="EMP404",
            amount=1000,
            claimed_category="meals",
            correct_category="miscellaneous",
            description="Personal grocery shopping - incorrectly classified",
            date_submitted=base_date + timedelta(days=10),
            date_of_expense=base_date + timedelta(days=10),
            has_gst_invoice=True,
            gst_invoice_valid=True,
            merchant_name="Supermarket",
            merchant_city="Mumbai",
            policy_compliant=False,
            is_fraud=True,
            fraud_types=[FraudType.PERSONAL_VS_BUSINESS],
            fraudulent_reason="Personal expense misclassified as business"
        ))
        
        # Duplicate claim
        claims.append(ExpenseClaim(
            employee_id="EMP401",
            amount=1200,
            claimed_category="travel",
            correct_category="travel",
            description="Duplicate flight booking claim - same trip",
            date_submitted=base_date + timedelta(days=11),
            date_of_expense=base_date + timedelta(days=6),
            has_gst_invoice=False,
            gst_invoice_valid=False,
            merchant_name="Air India",
            merchant_city="New Delhi",
            policy_compliant=False,
            is_fraud=True,
            fraud_types=[FraudType.DUPLICATE_CLAIM],
            fraudulent_reason="Duplicate claim for same flight"
        ))
        
        return claims

    def _generate_hard_claims(self) -> List[ExpenseClaim]:
        """Generate 15-18 claims with complex fraud patterns."""
        claims = self._generate_medium_claims()
        base_date = datetime.now() - timedelta(days=30)
        
        # Inflated amount
        claims.append(ExpenseClaim(
            employee_id="EMP501",
            amount=5000,
            claimed_category="travel",
            correct_category="travel",
            description="Accommodation - inflated cost",
            date_submitted=base_date + timedelta(days=15),
            date_of_expense=base_date + timedelta(days=12),
            has_gst_invoice=True,
            gst_invoice_valid=True,
            merchant_name="Hotel",
            merchant_city="Goa",
            policy_compliant=False,
            is_fraud=True,
            fraud_types=[FraudType.INFLATED_AMOUNT],
            fraudulent_reason="Amount significantly higher than hotel rate",
            metadata={"typical_rate": 2000, "claimed_rate": 5000}
        ))
        
        # Same-day round trip - suspicious
        claims.append(ExpenseClaim(
            employee_id="EMP502",
            amount=3500,
            claimed_category="travel",
            correct_category="travel",
            description="Mumbai to Pune same-day trip",
            date_submitted=base_date + timedelta(days=16),
            date_of_expense=base_date + timedelta(days=13),
            has_gst_invoice=False,
            gst_invoice_valid=False,
            merchant_name="Bus Service",
            merchant_city="Pune",
            mileage_claimed=300,
            policy_compliant=False,
            is_fraud=True,
            fraud_types=[FraudType.SAME_DAY_ROUND_TRIP],
            fraudulent_reason="Suspicious same-day round trip with high mileage claims"
        ))
        
        # Serial claim pattern - same employee multiple claims in short time
        for i in range(3):
            claims.append(ExpenseClaim(
                employee_id="EMP503",
                amount=800 + i*100,
                claimed_category="meals",
                correct_category="meals",
                description=f"Daily meal claim #{i+1}",
                date_submitted=base_date + timedelta(days=17+i),
                date_of_expense=base_date + timedelta(days=17+i),
                has_gst_invoice=i % 2 == 0,
                gst_invoice_valid=i % 2 == 0,
                merchant_name="Restaurant",
                merchant_city="Mumbai",
                policy_compliant=True if i == 0 else False,
                is_fraud=i > 0,
                fraud_types=[FraudType.SERIAL_CLAIM_PATTERN] if i > 0 else [],
                fraudulent_reason="Serial claim pattern - multiple claims same employee" if i > 0 else None
            ))
        
        # Mismatched dates
        claims.append(ExpenseClaim(
            employee_id="EMP504",
            amount=2000,
            claimed_category="travel",
            correct_category="travel",
            description="Travel claim with mismatched dates",
            date_submitted=base_date + timedelta(days=20),
            date_of_expense=base_date + timedelta(days=-5),  # Submitted after date of expense
            has_gst_invoice=True,
            gst_invoice_valid=False,
            merchant_name="Transport",
            merchant_city="Delhi",
            policy_compliant=False,
            is_fraud=True,
            fraud_types=[FraudType.MISMATCHED_DATES],
            fraudulent_reason="Submitted date before expense date"
        ))
        
        return claims
    
    # ============ OpenEnv Async API ============
    async def async_reset(self) -> 'StepResult':
        """Async reset - returns StepResult with Observation."""
        from test.models import StepResult, Observation
        state_dict = self.reset()
        obs = Observation(state=state_dict, info={})
        return StepResult(observation=obs, reward=0.0, done=False, info={})
    
    async def async_step(self, action: Dict[str, Any]) -> 'StepResult':
        """Async step - accepts dict action, returns StepResult."""
        from test.models import StepResult, Observation
        state_dict, reward, done, info = self.step(action)
        obs = Observation(state=state_dict, info=info)
        return StepResult(observation=obs, reward=reward, done=done, info=info)
