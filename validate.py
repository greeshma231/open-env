#!/usr/bin/env python3
"""Validation script for CorpExpenseAudit project."""

import sys
import traceback

def run_validations():
    """Run all validation checks."""
    try:
        print("="*70)
        print("CorpExpenseAudit - Project Validation")
        print("="*70)
        
        # Test 1: Import models
        print("\n[1/7] Testing model imports...")
        from test.models import ExpenseClaim, AuditState, ClaimCategory
        print("✓ Models loaded successfully")
        
        # Test 2: Import environment
        print("\n[2/7] Testing environment class...")
        from test.environment import CorpExpenseAudit
        print("✓ Environment class loaded")
        
        # Test 3: Import graders
        print("\n[3/7] Testing graders...")
        from test.graders import TaskGrader, run_easy_grader, run_medium_grader, run_hard_grader
        print("✓ Graders loaded")
        
        # Test 4: Initialize environment
        print("\n[4/7] Testing environment initialization (easy)...")
        env_easy = CorpExpenseAudit(task_difficulty='easy')
        state_easy = env_easy.reset()
        print("✓ Easy environment initialized")
        print(f"  - Total claims: {state_easy['total_claims']}")
        print(f"  - Max steps: {state_easy['max_steps']}")
        print(f"  - Pending claims: {len(state_easy['pending_claims'])}")
        
        # Test 5: Execute step
        print("\n[5/7] Testing step execution...")
        claim_id = state_easy['pending_claims'][0]
        action = {
            'action_type': 'inspect_claim',
            'action_data': {'claim_id': claim_id}
        }
        new_state, reward, done, info = env_easy.step(action)
        print("✓ Step executed successfully")
        print(f"  - Reward: {reward:+.4f}")
        print(f"  - Done: {done}")
        print(f"  - Current step: {env_easy.state.current_step}")
        
        # Test 6: Run grader
        print("\n[6/7] Testing grader...")
        metrics = run_easy_grader(env_easy)
        print("✓ Easy grader executed")
        print(f"  - Score: {metrics.final_score:.4f}")
        print(f"  - Efficiency: {metrics.efficiency_score:.2%}")
        
        # Test 7: Medium and Hard environments
        print("\n[7/7] Testing medium and hard environments...")
        env_medium = CorpExpenseAudit(task_difficulty='medium')
        state_medium = env_medium.reset()
        print(f"✓ Medium environment: {state_medium['total_claims']} claims")
        
        env_hard = CorpExpenseAudit(task_difficulty='hard')
        state_hard = env_hard.reset()
        print(f"✓ Hard environment: {state_hard['total_claims']} claims")
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print("✓ All imports successful")
        print("✓ Environment initialization works")
        print("✓ Step execution works")
        print("✓ Grading system works")
        print("✓ All 3 difficulty levels load correctly")
        print("\n✓✓✓ PROJECT VALIDATION PASSED ✓✓✓")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_validations()
    sys.exit(0 if success else 1)
