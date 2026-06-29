from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Mapping, Callable, Type, TypeVar, Generic, Union
import time
import hashlib
import json

# LangGraph imports with graceful fallback
try:  # pragma: no cover - soft dependency
    from langgraph.graph import StateGraph, START, END  # type: ignore
    from langgraph.graph.state import CompiledStateGraph  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    StateGraph = object  # type: ignore
    START = "START"
    END = "END"
    CompiledStateGraph = object  # type: ignore
    _LANGGRAPH_AVAILABLE = False

# Type variables for generic agent interfaces
TRequest = TypeVar('TRequest')
TResponse = TypeVar('TResponse')
TState = TypeVar('TState')

###############################################################################
# Data Models
###############################################################################


@dataclass(slots=True)
class AgentError:
    """Represents an error encountered during agent execution."""
    code: str
    message: str
    severity: str = "ERROR"  # INFO | WARNING | ERROR
    details: Dict[str, Any] = field(default_factory=dict)

    def checksum_material(self) -> str:
        return f"{self.severity}|{self.code}|{self.message}|{json.dumps(self.details, sort_keys=True)}"


@dataclass(slots=True)
class TimingInfo:
    """Execution timing information."""
    started_epoch_ms: int
    completed_epoch_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.completed_epoch_ms - self.started_epoch_ms)


@dataclass(slots=True)
class IntegrityInfo:
    """Checksum for integrity verification."""
    checksum: str
    algorithm: str = "SHA256"


@dataclass(slots=True)
class AgentResult:
    """Structured result from agent execution."""
    success: bool
    run_id: str
    rules_version: str
    timing: TimingInfo
    errors: List[AgentError] = field(default_factory=list)
    provenance_id: Optional[str] = None
    integrity: Optional[IntegrityInfo] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "run_id": self.run_id,
            "rules_version": self.rules_version,
            "timing": {
                "started_epoch_ms": self.timing.started_epoch_ms,
                "completed_epoch_ms": self.timing.completed_epoch_ms,
                "duration_ms": self.timing.duration_ms,
            },
            "errors": [asdict(e) for e in self.errors],
            "provenance_id": self.provenance_id,
            "integrity": asdict(self.integrity) if self.integrity else None,
            "payload": self.payload,
        }


@dataclass(slots=True)
class AgentContext:
    """Runtime context for agent execution."""
    run_id: str
    rules_version: str
    input_data: Mapping[str, Any] | None = None
    config: Mapping[str, Any] | None = None
    environment: Mapping[str, Any] | None = None
    provenance_id: Optional[str] = None
    llm: Any | None = None  # Injected LLM adapter
    adapters: Dict[str, Any] = field(default_factory=dict)
    workflow_state: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LangGraphConfig:
    """Configuration for LangGraph integration."""
    enable_langgraph: bool = True
    state_schema: Optional[Type] = None
    checkpointer: Optional[Any] = None
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
    debug: bool = False


def _stable_hash(parts: List[str], algorithm: str = "SHA256") -> IntegrityInfo:
    """Generate a stable hash from ordered parts."""
    h = hashlib.new(algorithm.lower())
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")  # field separator
    return IntegrityInfo(checksum=h.hexdigest(), algorithm=algorithm.upper())


###############################################################################
# Agent Base Classes
###############################################################################


class Agent(ABC):
    """Abstract base for all agents with LangGraph support.

    Implementations should:
      * Be deterministic given identical inputs & config.
      * Populate checksum over ordered logical components.
      * Avoid network / side-effects unless explicitly authorized.
      * Support LangGraph workflow execution patterns.
    """

    name: str = "agent"
    langgraph_enabled: bool = True
    _adapters: Dict[str, Any]
    _langgraph_config: Optional[LangGraphConfig]

    def __init__(self, name: Optional[str] = None, langgraph_config: Optional[LangGraphConfig] = None):
        if name:
            self.name = name
        self._langgraph_config = langgraph_config or LangGraphConfig()
        self._adapters = {}

    def _pre_run(self) -> int:
        return int(time.time() * 1000)

    def _post_run(self) -> int:
        return int(time.time() * 1000)

    def _result(
        self,
        ctx: AgentContext,
        success: bool,
        started: int,
        completed: int,
        payload: Dict[str, Any],
        errors: Optional[List[AgentError]] = None,
        checksum_parts: Optional[List[str]] = None,
    ) -> AgentResult:
        integrity = None
        if checksum_parts:
            integrity = _stable_hash(checksum_parts)
        return AgentResult(
            success=success,
            run_id=ctx.run_id,
            rules_version=ctx.rules_version,
            timing=TimingInfo(started_epoch_ms=started, completed_epoch_ms=completed),
            errors=errors or [],
            provenance_id=ctx.provenance_id,
            integrity=integrity,
            payload=payload,
        )

    # -------------------------------------------------------------------------
    # Adapter Integration
    # -------------------------------------------------------------------------
    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register a named adapter instance for use by this agent."""
        self._adapters[name] = adapter

    def get_adapter(self, name: str) -> Optional[Any]:
        """Retrieve a registered adapter by name."""
        return self._adapters.get(name)

    def get_llm_adapter(self) -> Optional[Any]:
        """Convenience method to get LLM adapter."""
        return self.get_adapter('llm')

    def get_agent_adapter(self) -> Optional[Any]:
        """Convenience method to get agent adapter."""
        return self.get_adapter('agent')

    # -------------------------------------------------------------------------
    # LangGraph Integration
    # -------------------------------------------------------------------------
    def supports_langgraph(self) -> bool:
        """Check if this agent supports LangGraph workflows."""
        return (
            self.langgraph_enabled
            and _LANGGRAPH_AVAILABLE
            and self._langgraph_config.enable_langgraph
        )

    def build_workflow(self, state_schema: Optional[Type] = None) -> Optional[Any]:
        """Build and compile a LangGraph workflow for this agent.

        Override this method in subclasses to define custom workflows.
        """
        if not self.supports_langgraph():
            return None
        return self._build_default_workflow(state_schema)

    def _build_default_workflow(self, state_schema: Optional[Type] = None) -> Optional[Any]:
        """Build a default single-node workflow."""
        if not _LANGGRAPH_AVAILABLE:
            return None

        from typing import TypedDict

        if state_schema is None:
            class DefaultState(TypedDict):
                messages: List[str]
                result: Dict[str, Any]
            state_schema = DefaultState

        workflow = StateGraph(state_schema)
        workflow.add_node("execute", self._default_execution_node)
        workflow.add_edge(START, "execute")
        workflow.add_edge("execute", END)

        return workflow.compile(
            checkpointer=self._langgraph_config.checkpointer,
            interrupt_before=self._langgraph_config.interrupt_before,
            interrupt_after=self._langgraph_config.interrupt_after,
            debug=self._langgraph_config.debug,
        )

    def _default_execution_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Default execution node for LangGraph workflows."""
        return {
            **state,
            "result": {"executed": True, "agent": self.name},
            "messages": state.get("messages", []) + [f"Executed {self.name}"]
        }

    def execute_with_langgraph(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute agent logic using LangGraph workflow."""
        if not self.supports_langgraph():
            raise NotImplementedError("LangGraph not available or not enabled")

        workflow = self.build_workflow()
        if workflow is None:
            raise ValueError("Failed to build LangGraph workflow")

        return workflow.invoke(initial_state, config=config)

    @abstractmethod
    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute the agent's main logic and return a structured AgentResult."""
        raise NotImplementedError


###############################################################################
# Specialized Agent Classes
###############################################################################


class LangGraphAgent(Agent, Generic[TState]):
    """Specialized agent base class for LangGraph-first implementations."""

    def __init__(self, name: Optional[str] = None, state_schema: Optional[Type[TState]] = None):
        super().__init__(name, LangGraphConfig(enable_langgraph=True))
        self.state_schema = state_schema

    @abstractmethod
    def build_workflow(self, state_schema: Optional[Type] = None) -> Optional[Any]:
        """Subclasses must implement their LangGraph workflow."""
        raise NotImplementedError

    def _build_initial_state(self, ctx: AgentContext, **kwargs) -> Dict[str, Any]:
        """Build initial state from context and kwargs."""
        return {
            "messages": [],
            "input_data": dict(ctx.input_data) if ctx.input_data else {},
            "config": dict(ctx.config) if ctx.config else {},
            **kwargs
        }

    def _extract_payload(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract payload from final workflow state."""
        return final_state.get("result", {})

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Default implementation using LangGraph execution."""
        started = self._pre_run()
        try:
            initial_state = self._build_initial_state(ctx, **kwargs)
            final_state = self.execute_with_langgraph(initial_state)
            payload = self._extract_payload(final_state)

            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
            )
        except Exception as e:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="WORKFLOW_ERROR", message=str(e))]
            )
