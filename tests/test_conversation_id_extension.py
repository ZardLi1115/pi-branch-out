from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extensions" / "tdai-conversation-id.ts"
AGENT = ROOT / "src" / "pi_branch_out" / "harbor_agent.py"


def _code_without_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_conversation_id_extension_rewrites_provider_headers() -> None:
    text = EXTENSION.read_text(encoding="utf-8")
    code = _code_without_comments(text)
    assert 'headers["x-conversation-id"]' in code
    assert 'pi.on("session_start"' in code
    assert 'pi.on("before_agent_start"' in code
    assert "pi.registerProvider" in code
    assert 'pi.on("before_provider_headers"' not in code


def test_harbor_agent_uploads_conversation_id_extension() -> None:
    text = AGENT.read_text(encoding="utf-8")
    code = _code_without_comments(text)
    assert "tdai-conversation-id.ts" in code
    assert "python3 - <<'PY'" in code
    assert '"apiKey": "CUSTOM_API_KEY"' in code
    assert "SUPPORTS_RESUME = True" in code
    assert "AgentCapabilities" not in code
    assert '"$CUSTOM_API_KEY"' not in code
