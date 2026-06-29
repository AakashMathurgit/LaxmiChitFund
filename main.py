"""
Main entry point for the LCF multi-agent system.
Handles configuration loading, agent creation, and pipeline orchestration.
"""

import os
from src.utils.logger import get_logger
from src.pipeline.orchestrator import PipelineOrchestrator


def main() -> None:
    """
    Main workflow for running the LCF pipeline.
    Delegates execution to the pipeline orchestrator.
    """
    logger = get_logger("Main")
    logger.info("Starting LCF pipeline...")

    # Run the main pipeline orchestrator
    orchestrator = PipelineOrchestrator()
    orchestrator.run()

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
