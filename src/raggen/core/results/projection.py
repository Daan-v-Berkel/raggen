from raggen.core.results.envelope import ResultEnvelope, ResultMessage
from enum import Enum


MAX_SUMMARY_WARNINGS = 3


def project_result(result: ResultEnvelope, detail: bool = False) -> ResultEnvelope:
    if detail:
        return result

    projected = result.model_copy(deep=True)

    # Only keep summary payload in summary mode.
    if isinstance(projected.data, dict):
        projected.data = {
            "summary": projected.data.get("summary")
        }

    # Truncate warnings.
    total_warnings = len(projected.warnings)
    if total_warnings > MAX_SUMMARY_WARNINGS:
        projected.warnings = projected.warnings[:MAX_SUMMARY_WARNINGS]
        projected.warnings.append(
            ResultMessage(
                code="WARNINGS_TRUNCATED",
                message=(
                    f"Showing {MAX_SUMMARY_WARNINGS} of {total_warnings} warnings. "
                    f"Inspect run '{projected.run_id}' with detail output to view all warnings."
                ),
            )
        )

    # Errors are kept in full.
    return projected
