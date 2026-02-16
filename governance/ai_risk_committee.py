from financial_organism.governance.go_live_checklist import build_context, evaluate_checklist
class AIRiskCommittee:
    def assert_go_live_ready(self):
        results = evaluate_checklist(build_context())
        failed = [x for x in results if not x["passed"]]
        if failed:
            reasons = "; ".join(f"{x['id']}: {x['description']}" for x in failed)
            raise RuntimeError(f"LIVE mode blocked by AI Risk Committee: {reasons}")
        return True
