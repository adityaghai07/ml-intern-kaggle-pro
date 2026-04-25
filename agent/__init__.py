"""
HF Agent - Main agent module
"""

import logging
import litellm

# Global LiteLLM behavior — set once at package import so both CLI and
# backend entries share the same config.
#   drop_params: quietly drop unsupported params rather than raising
#   suppress_debug_info: hide the noisy "Give Feedback" banner on errors
#   modify_params: let LiteLLM patch Anthropic's tool-call requirements
#     (synthesize a dummy tool spec when we call completion on a history
#     that contains tool_calls but aren't passing `tools=` — happens
#     during summarization / session seeding).
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.modify_params = True


# Suppress the "Dropping 'thinking' param" warning from LiteLLM's transformation module.
# This warning occurs when modify_params=True and the last assistant message with tool_calls
# has no thinking_blocks. The warning is informational and expected behavior when using
# tool calls without extended thinking, so we filter it out to reduce noise.
class LiteLLMThinkingWarningFilter(logging.Filter):
    """Filter to suppress LiteLLM's thinking parameter drop warnings."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress the specific warning about dropping thinking param
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            if "Dropping 'thinking' param" in msg and "thinking_blocks" in msg:
                return False
        return True


# Apply the filter to the LiteLLM logger
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.addFilter(LiteLLMThinkingWarningFilter())

from agent.core.agent_loop import submission_loop  # noqa: E402

__all__ = ["submission_loop"]
