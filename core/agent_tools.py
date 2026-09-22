# -*- coding: utf-8 -*-
"""Tool registry for the BlindGuard agent.

The registry keeps tool declarations, execution policy, validation, timeout
handling, result normalization and recent execution audit in one place.  The
agent remains responsible for the actual tool handlers and data sources.

执行模型（安全相关）：
- 只读工具：在 daemon 线程中带超时执行，超时返回"已超时，结果不可用"，
  不阻塞对话。
- 写工具（access == "write"）：同步执行，绝不使用"超时即放弃但线程继续
  跑完"的模型，保证"报告结果"与"真实副作用"严格一致（例如
  set_user_preference 不会表面报超时失败、实际仍把偏好写入磁盘）。
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("BlindGuard.Agent.Tools")


class ToolExecutionError(Exception):
    """Raised when a tool cannot be executed safely."""


class ToolTimeoutError(ToolExecutionError):
    """结构化超时错误：不再依赖子串匹配判断是否超时。"""


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by a registered tool."""

    name: str
    success: bool
    payload: Dict[str, Any]
    content: str
    duration_ms: int
    error: str = ""
    timed_out: bool = False


class ToolRegistry:
    """Register, validate and execute agent tools with bounded execution."""

    def __init__(self, result_limit: int = 8000):
        self.result_limit = max(1000, int(result_limit))
        self._specs: Dict[str, Dict[str, Any]] = {}
        self._audit: List[Dict[str, Any]] = []

    def register(
        self,
        definition: Dict[str, Any],
        handler: Callable[..., Any],
        *,
        access: str = "read",
        safety_level: str = "standard",
        timeout_ms: int = 2500,
    ) -> None:
        """Register one OpenAI-compatible function declaration."""
        fn = (definition or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            raise ValueError("tool name is required")
        if name in self._specs:
            raise ValueError(f"duplicate tool: {name}")
        if not callable(handler):
            raise TypeError(f"handler for {name} must be callable")
        if access not in ("read", "write"):
            raise ValueError(f"invalid access mode for {name}: {access}")

        self._specs[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(fn.get("description") or ""),
                    "parameters": fn.get("parameters") or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            "handler": handler,
            "access": access,
            "safety_level": safety_level,
            "timeout_ms": max(50, int(timeout_ms or 2500)),
        }

    @property
    def names(self) -> List[str]:
        return list(self._specs)

    def openai_tools(self) -> List[Dict[str, Any]]:
        """Return fresh schema dictionaries safe for request serialization."""
        return [json.loads(json.dumps(item["definition"], ensure_ascii=False))
                for item in self._specs.values()]

    def capabilities(self) -> List[Dict[str, Any]]:
        """Return compact tool metadata for health/capability reporting."""
        result = []
        for name, item in self._specs.items():
            result.append({
                "name": name,
                "description": item["definition"]["function"]["description"],
                "access": item["access"],
                "safety_level": item["safety_level"],
                "timeout_ms": item["timeout_ms"],
            })
        return result

    def metadata(self, name: str) -> Dict[str, Any]:
        """Return one tool's execution policy without exposing its handler."""
        item = self._specs.get(str(name or ""))
        if item is None:
            return {}
        return {
            "name": str(name),
            "access": item["access"],
            "safety_level": item["safety_level"],
            "timeout_ms": item["timeout_ms"],
        }

    def execute(self, name: str, arguments: Any) -> ToolResult:
        """Validate and execute one tool call, always returning a result."""
        started = time.perf_counter()
        spec = self._specs.get(str(name or ""))
        if spec is None:
            return self._result(
                name=str(name or ""),
                success=False,
                payload={"error": f"未知工具: {name}"},
                error=f"unknown tool: {name}",
                started=started,
            )

        try:
            args = self._parse_arguments(arguments)
            self._validate_arguments(name, spec["definition"], args)
            payload = self._invoke_tool(
                spec["handler"], args, spec["timeout_ms"], spec["access"])
            return self._result(
                name=name,
                success=True,
                payload=self._normalize_payload(payload),
                started=started,
            )
        except ToolExecutionError as exc:
            timed_out = isinstance(exc, ToolTimeoutError)
            logger.warning("工具 %s 执行失败: %s", name, exc)
            return self._result(
                name=name,
                success=False,
                payload={"error": str(exc)},
                error=str(exc),
                started=started,
                timed_out=timed_out,
            )
        except Exception as exc:
            logger.exception("工具 %s 执行异常", name)
            return self._result(
                name=name,
                success=False,
                payload={"error": str(exc)},
                error=f"{type(exc).__name__}: {exc}",
                started=started,
            )

    def recent_audit(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self._audit[-max(0, int(limit)):]

    def clear_audit(self) -> None:
        self._audit.clear()

    @staticmethod
    def _parse_arguments(arguments: Any) -> Dict[str, Any]:
        if arguments in (None, ""):
            return {}
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolExecutionError(f"参数不是合法 JSON: {exc.msg}") from exc
            if not isinstance(parsed, dict):
                raise ToolExecutionError("工具参数必须是 JSON 对象")
            return parsed
        raise ToolExecutionError("工具参数必须是 JSON 对象")

    @staticmethod
    def _validate_arguments(
        name: str,
        definition: Dict[str, Any],
        args: Dict[str, Any],
    ) -> None:
        schema = definition["function"].get("parameters") or {}
        for key in schema.get("required") or []:
            if key not in args or args[key] in (None, ""):
                raise ToolExecutionError(f"{name} 缺少必填参数: {key}")

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, prop in (schema.get("properties") or {}).items():
            if key not in args:
                continue
            expected = prop.get("type")
            if expected in ("number", "integer") and isinstance(args[key], bool):
                raise ToolExecutionError(f"{name}.{key} 必须是数字而非布尔值")
            if expected in type_map and not isinstance(args[key], type_map[expected]):
                raise ToolExecutionError(
                    f"{name}.{key} 类型应为 {expected}"
                )
            allowed = prop.get("enum")
            if allowed and args[key] not in allowed:
                raise ToolExecutionError(
                    f"{name}.{key} 取值不在允许范围内"
                )

    @classmethod
    def _invoke_tool(
        cls,
        handler: Callable[..., Any],
        args: Dict[str, Any],
        timeout_ms: int,
        access: str,
    ) -> Any:
        """按访问模式分发执行，保证"报告结果"与"真实副作用"一致。

        写工具（access == "write"，如 set_user_preference）有副作用，必须
        同步执行：要么成功、要么抛错，绝不出现"表面报超时失败、后台线程
        却仍把副作用落盘"的认知失真。只读工具才使用带超时的线程模型。
        """
        if access == "write":
            # 同步执行写工具：结果与副作用严格一致，不可"报失败仍生效"。
            return handler(**dict(args or {}))
        return cls._invoke_with_timeout(handler, args, timeout_ms)

    @staticmethod
    def _invoke_with_timeout(
        handler: Callable[..., Any],
        args: Dict[str, Any],
        timeout_ms: int,
    ) -> Any:
        """Run a tool in a daemon thread so a hung read call cannot block chat."""
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)

        def run():
            try:
                result_queue.put_nowait(("ok", handler(**dict(args or {}))))
            except Exception as exc:  # re-raised in the caller thread
                result_queue.put_nowait(("error", exc))

        worker = threading.Thread(
            target=run,
            daemon=True,
            name=f"AgentTool-{getattr(handler, '__name__', 'call')}",
        )
        worker.start()
        try:
            status, value = result_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty as exc:
            raise ToolTimeoutError(
                f"工具执行超时（{timeout_ms}ms），已超时，结果不可用"
            ) from exc
        if status == "error":
            if isinstance(value, Exception):
                raise value
            raise ToolExecutionError(str(value))
        return value

    def _normalize_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
        if payload is None:
            return {"result": None}
        return {"result": str(payload)}

    def _result(
        self,
        *,
        name: str,
        success: bool,
        payload: Dict[str, Any],
        started: float,
        error: str = "",
        timed_out: bool = False,
    ) -> ToolResult:
        duration_ms = max(0, round((time.perf_counter() - started) * 1000))
        try:
            content = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            content = json.dumps({"error": "工具结果无法序列化"},
                                 ensure_ascii=False)
        if len(content) > self.result_limit:
            content = content[:self.result_limit] + "...(结果已截断)"
            payload = dict(payload)
            payload["_truncated"] = True
        result = ToolResult(
            name=name,
            success=success,
            payload=payload,
            content=content,
            duration_ms=duration_ms,
            error=error,
            timed_out=timed_out,
        )
        self._audit.append({
            "time": round(time.time(), 3),
            "name": name,
            "success": success,
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "error": error[:200],
        })
        self._audit = self._audit[-100:]
        return result
