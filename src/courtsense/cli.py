"""Command-line entry point for the M1 demonstration."""

import argparse

from courtsense.domain.models import ConsentRecord, Session
from courtsense.pipeline.runner import run_synthetic_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline CourtSense prototype")
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--participant-id", default="demo-participant")
    parser.add_argument(
        "--retention",
        choices=("none", "history", "training"),
        default="none",
    )
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--tolerance-ns", type=int, default=2_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    consent = ConsentRecord(
        participant_id=args.participant_id,
        history_retention_allowed=args.retention in {"history", "training"},
        training_retention_allowed=args.retention == "training",
    )
    result = run_synthetic_analysis(
        Session(args.session_id, args.participant_id),
        consent,
        frame_count=args.frame_count,
        timestamp_tolerance_ns=args.tolerance_ns,
    )
    print(f"session={result.session.session_id}")
    print(f"retention={result.retention_mode.value}")
    print(f"provenance={result.sequence.provenance.value}")
    print(f"paired_frames={len(result.synchronization.pairs)}")
    print(f"quality={result.quality.status.value}")
    for feature in result.features:
        print(f"{feature.feature_name}={feature.value} {feature.units}")
    print(f"retained_artifacts={len(result.retained_artifact_ids)}")


if __name__ == "__main__":
    main()
