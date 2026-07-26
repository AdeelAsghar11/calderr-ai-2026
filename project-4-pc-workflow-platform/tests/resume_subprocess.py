"""
Subprocess script used to test cross-process persistence resumption.
Executed by test_phase1.py in a separate OS process.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import WorkflowEngine


def main():
    if len(sys.argv) < 4:
        print("Usage: python resume_subprocess.py <db_path> <thread_id> <resume_value>")
        sys.exit(1)

    db_path = sys.argv[1]
    thread_id = sys.argv[2]
    resume_value = sys.argv[3]

    engine = WorkflowEngine(db_path=db_path)
    engine.register_yaml_file("project-4-pc-workflow-platform/workflows/2_human_approval.yaml")
    
    result = engine.resume_workflow("human_approval", thread_id, resume_value)
    print(f"RESUME_SUCCESS status={result['status']} state={result['state']}")


if __name__ == "__main__":
    main()
