from raggen.core.results.envelope import ResultEnvelope, ResultMessage

MAX_SUMMARY_WARNINGS = 3


def project_result(result: ResultEnvelope, detailed: bool = False) -> ResultEnvelope:
    projected = result.model_copy(deep=True)

    if detailed:
        if isinstance(projected.data, dict):
            projected.data = projected.data.get("details")
            return projected

    # Only keep summary payload in summary mode.
    if isinstance(projected.data, dict):
        projected.data = projected.data.get("summary")

    # Truncate warnings.
    total_warnings = len(projected.warnings)
    if total_warnings > MAX_SUMMARY_WARNINGS:
        projected.warnings = projected.warnings[:MAX_SUMMARY_WARNINGS]
        projected.warnings.append(
            ResultMessage(
                code="WARNINGS_TRUNCATED",
                message=(
                    f"Showing {MAX_SUMMARY_WARNINGS} of {total_warnings} warnings. "
                    f"Inspect run '{projected.run_id}' with --detailed to view complete output."
                ),
            )
        )

    # Errors are kept in full.
    return projected
