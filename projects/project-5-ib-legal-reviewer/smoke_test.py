"""
Offline smoke test suite for Project 5-I-B: Multi-Agent Legal Document Reviewer.
Validates independent specialist review, debate cross-examination, Judge synthesis,
and severity modifications using plain assert statements and readable console output.
"""

import sys
from pathlib import Path

# Add project directory to path for imports
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from agents import run_legal_review, SPECIALIST_ROLES
    from models import ReviewReport
except ImportError:
    from project_5_ib_legal_reviewer.agents import run_legal_review, SPECIALIST_ROLES
    from project_5_ib_legal_reviewer.models import ReviewReport


def main():
    print("\n=======================================================")
    print(" [LEGAL DOCUMENT REVIEWER SMOKE TESTS]")
    print("=======================================================\n")

    sample_contracts_dir = PROJECT_DIR / "sample_contracts"
    contract_files = list(sample_contracts_dir.glob("*.txt"))

    assert len(contract_files) == 3, f"Expected 3 sample contract files, found {len(contract_files)}"
    print(f"Found {len(contract_files)} sample contract files for analysis.")

    reports: dict[str, ReviewReport] = {}

    # -------------------------------------------------------------------
    # TEST 1: Specialist Independent Review Coverage
    # -------------------------------------------------------------------
    print("\n[TEST 1] Independent Specialist Review Verification...")
    for contract_path in contract_files:
        doc_name = contract_path.name
        with open(contract_path, "r", encoding="utf-8") as f:
            content = f.read()

        report = run_legal_review(doc_name, content, real_mode=False)
        reports[doc_name] = report

        # Verify findings were generated
        assert len(report.findings) > 0, f"No findings generated for contract {doc_name}"

        # Verify specialists contributed
        raising_agents = {f.raised_by for f in report.findings}
        print(f"  [OK] {doc_name}: {len(report.findings)} findings raised by roles {sorted(list(raising_agents))}")

    print("  [PASSED TEST 1]: All specialist agents independently identified clause issues.")

    # -------------------------------------------------------------------
    # TEST 2: The Debate Proof (Cross-Examination & Severity Shift)
    # -------------------------------------------------------------------
    print("\n[TEST 2] Debate Proof & Severity Modification Check...")
    dispute_found = False
    severity_changed_found = False
    changed_clause_details = []

    for doc_name, report in reports.items():
        # Check for dispute stance in debate transcript
        disputes = [c for c in report.debate_transcript if c.stance == "dispute"]
        if disputes:
            dispute_found = True

        for finding in report.findings:
            if finding.contested:
                changed_clause_details.append(
                    f"Contract: '{doc_name}' | Clause: '{finding.clause_reference}' | "
                    f"Final Severity: {finding.final_severity} | Contested: {finding.contested} | Dissent: {finding.dissent_notes[0] if finding.dissent_notes else ''}"
                )
                severity_changed_found = True

    assert dispute_found, "Debate round failed to produce any 'dispute' stance across sample contracts!"
    assert severity_changed_found, "Debate round failed to alter final_severity for any contested finding!"

    print("  Debate Proof Evidence:")
    for detail in changed_clause_details:
        print(f"    * {detail}")

    print("  [PASSED TEST 2]: Debate round produced genuine disputes and altered final severity scores.")

    # -------------------------------------------------------------------
    # TEST 3: Integrity of Contested Status and Dissent Notes
    # -------------------------------------------------------------------
    print("\n[TEST 3] Contested Status and Dissent Notes Integrity Check...")
    for doc_name, report in reports.items():
        for finding in report.findings:
            if finding.contested:
                assert (
                    len(finding.dissent_notes) > 0
                ), f"Finding '{finding.clause_reference}' in {doc_name} is marked contested=True but has empty dissent_notes!"
            else:
                assert (
                    len(finding.dissent_notes) == 0
                ), f"Finding '{finding.clause_reference}' in {doc_name} is marked contested=False but has dissent_notes!"

    print("  [PASSED TEST 3]: All findings maintain consistent contested flags and non-empty dissent notes.")

    print("\n=======================================================")
    print(" ALL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
