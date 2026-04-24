"""
Kaggle Tool — Browse competitions, research notebooks/discussions, track scores, and submit.

Operations:
  list_competitions, competition_details, list_data_files,
  list_notebooks, read_notebook, list_discussions, read_discussion,
  leaderboard, my_submissions, submit, score_history

Auth: Supports both new-style Bearer tokens (KAGGLE_KEY starting with "KGAT_")
and legacy Basic auth (KAGGLE_USERNAME + KAGGLE_KEY).
Read operations use httpx directly (async-native).
Submit uses the kaggle Python package via asyncio.to_thread().
Score tracking persists to ~/.kaggle/agent_scores/{competition}.json.
"""

import asyncio
import json
import logging
import os
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

import httpx

from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)

KAGGLE_API = "https://www.kaggle.com/api/v1"
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_NOTEBOOK_CHARS = 12_000

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _kaggle_auth_header() -> dict[str, str]:
    """Return auth header from env vars.

    Supports two formats:
    - New-style API token: KAGGLE_KEY starts with "KGAT_" → Bearer auth
    - Legacy: KAGGLE_USERNAME + KAGGLE_KEY → Basic auth
    """
    key = os.environ.get("KAGGLE_KEY", "")
    if not key:
        return {}
    # New-style token (starts with KGAT_) — use Bearer auth
    if key.startswith("KGAT_"):
        return {"Authorization": f"Bearer {key}"}
    # Legacy format — needs username for Basic auth
    username = os.environ.get("KAGGLE_USERNAME", "")
    if not username:
        return {}
    token = b64encode(f"{username}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _require_auth() -> dict[str, str]:
    """Return auth header or raise if credentials are missing."""
    headers = _kaggle_auth_header()
    if not headers:
        raise ValueError(
            "Kaggle credentials not found. Set KAGGLE_KEY (new-style KGAT_ token) "
            "or KAGGLE_USERNAME + KAGGLE_KEY (legacy) environment variables. "
            "Get them from https://www.kaggle.com/settings → API."
        )
    return headers


# ---------------------------------------------------------------------------
# Score persistence
# ---------------------------------------------------------------------------

_SCORES_DIR = Path.home() / ".kaggle" / "agent_scores"


def _scores_path(competition: str) -> Path:
    _SCORES_DIR.mkdir(parents=True, exist_ok=True)
    return _SCORES_DIR / f"{competition}.json"


def _load_scores(competition: str) -> list[dict]:
    path = _scores_path(competition)
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_score(competition: str, entry: dict) -> None:
    scores = _load_scores(competition)
    scores.append(entry)
    _scores_path(competition).write_text(json.dumps(scores, indent=2))


# ---------------------------------------------------------------------------
# Run log persistence (cross-session learning)
# ---------------------------------------------------------------------------

_RUNS_DIR = Path.home() / ".kaggle" / "agent_runs"


def _runs_path(competition: str) -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return _RUNS_DIR / f"{competition}.json"


def _load_runs(competition: str) -> list[dict]:
    path = _runs_path(competition)
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_run(competition: str, entry: dict) -> None:
    runs = _load_runs(competition)
    entry.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    runs.append(entry)
    _runs_path(competition).write_text(json.dumps(runs, indent=2))


def _today_submission_count(competition: str) -> int:
    """Count submissions made today (for daily cap enforcement)."""
    today = time.strftime("%Y-%m-%d")
    runs = _load_runs(competition)
    return sum(1 for r in runs if r.get("type") == "submission" and r.get("timestamp", "").startswith(today))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _kaggle_get(
    client: httpx.AsyncClient,
    path: str,
    params: dict | None = None,
) -> dict | list | None:
    """GET from Kaggle API with auth + retries."""
    headers = _require_auth()
    url = f"{KAGGLE_API}{path}"
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=headers, params=params or {}, timeout=15)
            if resp.status_code == 401:
                raise ValueError("Kaggle auth failed — check KAGGLE_USERNAME and KAGGLE_KEY.")
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                return None
            if resp.status_code >= 500:
                if attempt < 2:
                    await asyncio.sleep(3)
                    continue
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError:
            if attempt < 2:
                await asyncio.sleep(3)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

async def _list_competitions(args: dict, limit: int) -> ToolResult:
    """List active Kaggle competitions."""
    params: dict[str, Any] = {"page": 1}
    if args.get("search"):
        params["search"] = args["search"]
    if args.get("category"):
        params["category"] = args["category"]
    if args.get("sort_by"):
        params["sortBy"] = args["sort_by"]

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, "/competitions/list", params)

    if data is None:
        return ToolResult(formatted="No competitions found or API error.", isError=True)

    comps = data[:limit] if isinstance(data, list) else []
    lines = [f"**Active Kaggle Competitions** ({len(comps)} shown)\n"]
    for c in comps:
        deadline = c.get("deadline", "")[:10] if c.get("deadline") else "N/A"
        reward = c.get("reward", "N/A")
        cat = c.get("category", "")
        lines.append(
            f"- **{c.get('title', c.get('ref', '?'))}** (`{c.get('ref', '')}`) "
            f"— {cat} | reward: {reward} | deadline: {deadline}"
        )
    return ToolResult(formatted="\n".join(lines), totalResults=len(comps), resultsShared=len(comps))


async def _competition_details(args: dict, _limit: int) -> ToolResult:
    """Get details for a single competition."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    async with httpx.AsyncClient() as client:
        # The list endpoint with search is the most reliable way to get details
        data = await _kaggle_get(client, "/competitions/list", {"search": comp})

    if not data or not isinstance(data, list):
        return ToolResult(formatted=f"Competition '{comp}' not found.", isError=True)

    # Find exact match
    match = None
    for c in data:
        if c.get("ref", "").split("/")[-1] == comp or c.get("ref") == comp:
            match = c
            break
    if not match and data:
        match = data[0]
    if not match:
        return ToolResult(formatted=f"Competition '{comp}' not found.", isError=True)

    deadline = match.get("deadline", "")[:10] if match.get("deadline") else "N/A"
    desc = match.get("description", "No description available.")
    lines = [
        f"# {match.get('title', comp)}",
        f"**Ref**: `{match.get('ref', comp)}`",
        f"**Category**: {match.get('category', 'N/A')}",
        f"**Reward**: {match.get('reward', 'N/A')}",
        f"**Deadline**: {deadline}",
        f"**Evaluation Metric**: {match.get('evaluationMetric', 'N/A')}",
        f"**Team Count**: {match.get('teamCount', 'N/A')}",
        f"**Max Daily Submissions**: {match.get('maxDailySubmissions', 'N/A')}",
        f"**Merger Deadline**: {match.get('mergerDeadline', 'N/A')[:10] if match.get('mergerDeadline') else 'N/A'}",
        "",
        f"**Description**: {desc[:1000]}",
        "",
        f"URL: https://www.kaggle.com/competitions/{match.get('ref', comp).split('/')[-1]}",
    ]
    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _list_data_files(args: dict, limit: int) -> ToolResult:
    """List data files for a competition."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, f"/competitions/data/list/{comp}")

    if data is None:
        return ToolResult(formatted=f"No data files found for '{comp}'.", isError=True)

    files = data[:limit] if isinstance(data, list) else []
    lines = [f"**Data files for `{comp}`** ({len(files)} files)\n"]
    for f in files:
        size_mb = f.get("totalBytes", 0) / (1024 * 1024)
        lines.append(f"- `{f.get('name', '?')}` — {size_mb:.1f} MB")
    return ToolResult(formatted="\n".join(lines), totalResults=len(files), resultsShared=len(files))


async def _list_notebooks(args: dict, limit: int) -> ToolResult:
    """List public notebooks (kernels) for a competition."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    sort_by = args.get("sort_by", "scoreAscending")
    params: dict[str, Any] = {
        "page": 1,
        "competition": comp,
        "sortBy": sort_by,
        "pageSize": min(limit, 20),
    }

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, "/kernels/list", params)

    if data is None:
        return ToolResult(formatted=f"No notebooks found for '{comp}'.", isError=True)

    notebooks = data[:limit] if isinstance(data, list) else []
    lines = [f"**Notebooks for `{comp}`** (sorted by {sort_by}, {len(notebooks)} shown)\n"]
    for nb in notebooks:
        score = nb.get("competitionScore", "—")
        votes = nb.get("totalVotes", 0)
        title = nb.get("title", "Untitled")
        author = nb.get("author", "?")
        ref = nb.get("ref", "")
        lines.append(f"- **{title}** by {author} — score: {score}, votes: {votes} (`{ref}`)")
    return ToolResult(formatted="\n".join(lines), totalResults=len(notebooks), resultsShared=len(notebooks))


async def _read_notebook(args: dict, _limit: int) -> ToolResult:
    """Read a notebook's source, render to markdown if .ipynb."""
    ref = args.get("notebook", "") or args.get("ref", "")
    if not ref:
        return ToolResult(formatted="Error: `notebook` parameter is required (e.g. 'owner/slug').", isError=True)

    # Pull kernel source via API
    params = {"userName": ref.split("/")[0], "kernelSlug": ref.split("/")[-1]} if "/" in ref else {"kernelSlug": ref}

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, "/kernels/pull", params)

    if data is None:
        return ToolResult(formatted=f"Notebook '{ref}' not found.", isError=True)

    source = data.get("blob", {}).get("source", "")
    language = data.get("blob", {}).get("language", "python")
    title = data.get("metadata", {}).get("title", ref)

    if not source:
        return ToolResult(formatted=f"Notebook '{ref}' has no source content.", isError=True)

    # Try to render as notebook if it looks like JSON/ipynb
    rendered = source
    if source.strip().startswith("{"):
        try:
            import nbformat
            from nbconvert import MarkdownExporter

            nb = nbformat.reads(source, as_version=4)
            exporter = MarkdownExporter()
            rendered, _ = exporter.from_notebook_node(nb)
        except Exception:
            pass  # Fall back to raw source

    # Truncate if too long
    if len(rendered) > MAX_NOTEBOOK_CHARS:
        rendered = rendered[:MAX_NOTEBOOK_CHARS] + f"\n\n... (truncated at {MAX_NOTEBOOK_CHARS} chars)"

    header = f"# {title}\n**Language**: {language} | **Ref**: `{ref}`\n\n"
    return ToolResult(formatted=header + rendered, totalResults=1, resultsShared=1)


async def _notebook_metadata(args: dict, _limit: int) -> ToolResult:
    """Get a notebook's metadata — data/model/kernel sources, GPU settings.

    Use this to copy the exact sources from a working notebook before pushing
    your own, avoiding wrong slugs, missing packages, etc.
    """
    ref = args.get("notebook", "") or args.get("ref", "")
    if not ref:
        return ToolResult(formatted="Error: `notebook` parameter is required (e.g. 'owner/slug').", isError=True)

    params = {"userName": ref.split("/")[0], "kernelSlug": ref.split("/")[-1]} if "/" in ref else {"kernelSlug": ref}

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, "/kernels/pull", params)

    if data is None:
        return ToolResult(formatted=f"Notebook '{ref}' not found.", isError=True)

    meta = data.get("metadata", {})
    machine_shape = meta.get("machineShape") or meta.get("machineShapeNullable") or ""
    docker_image = meta.get("dockerImage") or meta.get("dockerImageNullable") or ""
    lines = [
        f"# Notebook metadata for `{ref}`\n",
        f"**Title**: {meta.get('title', '?')}",
        f"**GPU**: {meta.get('enableGpu', False)}",
        f"**TPU**: {meta.get('enableTpu', False)}",
        f"**Internet**: {meta.get('enableInternet', False)}",
        f"**Accelerator (machineShape)**: {machine_shape or 'default'}",
        f"**Docker image**: {docker_image or 'default'}",
        "",
        "## Sources (use these exact values in push_notebook):",
        f"**competition_sources**: {meta.get('competitionDataSources', [])}",
        f"**dataset_sources**: {meta.get('datasetDataSources', [])}",
        f"**model_sources**: {meta.get('modelDataSources', [])}",
        f"**kernel_sources**: {meta.get('kernelDataSources', [])}",
    ]
    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _list_discussions(args: dict, limit: int) -> ToolResult:
    """List discussion topics for a competition forum."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    # Kaggle forums API — may not be publicly documented; graceful fallback
    params: dict[str, Any] = {
        "page": 1,
        "forumId": "",
        "sort": "recent",
    }

    async with httpx.AsyncClient() as client:
        # Try the competition-specific forum listing
        data = await _kaggle_get(
            client,
            f"/competitions/{comp}/forum/list",
            params,
        )
        # Fallback to general forum API if competition-specific fails
        if data is None:
            data = await _kaggle_get(
                client,
                "/forums/list",
                {"competition": comp, "page": 1},
            )

    if data is None or (isinstance(data, list) and len(data) == 0):
        return ToolResult(
            formatted=f"No discussions found for '{comp}'. "
            f"Browse manually: https://www.kaggle.com/competitions/{comp}/discussion",
        )

    topics = data[:limit] if isinstance(data, list) else []
    lines = [f"**Discussions for `{comp}`** ({len(topics)} shown)\n"]
    for t in topics:
        title = t.get("title", "Untitled")
        votes = t.get("voteCount", 0)
        comments = t.get("commentCount", 0)
        topic_id = t.get("id", "")
        lines.append(f"- **{title}** — {votes} votes, {comments} comments (id: {topic_id})")
    return ToolResult(formatted="\n".join(lines), totalResults=len(topics), resultsShared=len(topics))


async def _read_discussion(args: dict, _limit: int) -> ToolResult:
    """Read a specific discussion topic."""
    topic_id = args.get("topic_id", "")
    if not topic_id:
        return ToolResult(formatted="Error: `topic_id` parameter is required.", isError=True)

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, f"/forums/{topic_id}")

    if data is None:
        return ToolResult(
            formatted=f"Discussion topic {topic_id} not found or forum API unavailable.",
            isError=True,
        )

    title = data.get("title", "Untitled")
    content = data.get("content", "No content.")
    author = data.get("author", "?")
    votes = data.get("voteCount", 0)

    # Truncate long discussions
    if len(content) > 8000:
        content = content[:8000] + "\n\n... (truncated)"

    lines = [
        f"# {title}",
        f"**Author**: {author} | **Votes**: {votes}",
        "",
        content,
    ]

    # Include replies if present
    comments = data.get("comments", [])
    if comments:
        lines.append(f"\n---\n**Replies** ({len(comments)}):\n")
        for c in comments[:10]:
            c_author = c.get("author", "?")
            c_content = c.get("content", "")[:500]
            c_votes = c.get("voteCount", 0)
            lines.append(f"**{c_author}** (+{c_votes}): {c_content}\n")

    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _leaderboard(args: dict, limit: int) -> ToolResult:
    """Get competition leaderboard."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, f"/competitions/{comp}/leaderboard/view")

    if data is None:
        return ToolResult(formatted=f"Leaderboard not available for '{comp}'.", isError=True)

    # API may return {"submissions": [...]} or a list directly
    entries = data if isinstance(data, list) else data.get("submissions", data.get("teams", []))
    entries = entries[:limit]

    lines = [f"**Leaderboard for `{comp}`** (top {len(entries)})\n"]
    lines.append("| Rank | Team/User | Score | Entries |")
    lines.append("|------|-----------|-------|---------|")
    for e in entries:
        rank = e.get("rank", e.get("teamRank", "?"))
        name = e.get("teamName", e.get("submitter", "?"))
        score = e.get("score", "?")
        n_entries = e.get("submissionCount", e.get("entries", "?"))
        lines.append(f"| {rank} | {name} | {score} | {n_entries} |")
    return ToolResult(formatted="\n".join(lines), totalResults=len(entries), resultsShared=len(entries))


async def _my_submissions(args: dict, limit: int) -> ToolResult:
    """List user's own submissions for a competition."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    async with httpx.AsyncClient() as client:
        data = await _kaggle_get(client, f"/competitions/submissions/list/{comp}")

    if data is None:
        return ToolResult(formatted=f"No submissions found for '{comp}'.", isError=True)

    subs = data[:limit] if isinstance(data, list) else []

    # Also load local score history for trajectory
    local_scores = _load_scores(comp)

    lines = [f"**My submissions for `{comp}`** ({len(subs)} shown)\n"]
    for s in subs:
        score = s.get("publicScore", s.get("score", "pending"))
        status = s.get("status", "?")
        date = s.get("date", "")[:19] if s.get("date") else ""
        desc = s.get("description", "")
        lines.append(f"- [{date}] score: **{score}** — status: {status} | {desc}")

    if local_scores:
        lines.append(f"\n**Local tracking**: {len(local_scores)} submissions tracked")
        best = min(local_scores, key=lambda x: float(x.get("score", "inf")) if x.get("score") not in (None, "", "pending") else float("inf"))
        if best.get("score") not in (None, "", "pending"):
            lines.append(f"Best local score: **{best['score']}** ({best.get('hypothesis', '')})")

    return ToolResult(formatted="\n".join(lines), totalResults=len(subs), resultsShared=len(subs))


async def _submit(args: dict, _limit: int) -> ToolResult:
    """Submit predictions to a Kaggle competition.

    Uses httpx multipart upload with Bearer/Basic auth.
    Falls back to kaggle Python package if direct upload fails.
    Records the submission in local score history.
    """
    comp = args.get("competition", "")
    file_path = args.get("file_path", "")
    message = args.get("message", "Agent submission")
    hypothesis = args.get("hypothesis", "")

    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)
    if not file_path:
        return ToolResult(formatted="Error: `file_path` parameter is required.", isError=True)
    if not os.path.isfile(file_path):
        return ToolResult(formatted=f"Error: file not found: {file_path}", isError=True)

    # Daily submission cap check
    today_count = _today_submission_count(comp)
    if today_count >= 3:
        return ToolResult(
            formatted=f"Daily submission cap reached ({today_count}/3 today). Save your remaining attempts for tomorrow. "
            f"Use this time to research better approaches instead.",
            isError=True,
        )

    headers = _require_auth()
    result = None

    # Try direct httpx upload first (works with both Bearer and Basic auth)
    try:
        async with httpx.AsyncClient() as client:
            url = f"{KAGGLE_API}/competitions/submissions/url/{comp}"
            # Step 1: get upload URL
            resp = await client.post(
                url,
                headers=headers,
                json={"fileName": os.path.basename(file_path), "contentLength": os.path.getsize(file_path)},
                timeout=30,
            )
            if resp.status_code == 200:
                upload_info = resp.json()
                upload_url = upload_info.get("createUrl", "")
                if upload_url:
                    # Step 2: upload file
                    with open(file_path, "rb") as f:
                        upload_resp = await client.put(upload_url, content=f.read(), timeout=120)
                    if upload_resp.status_code in (200, 201):
                        # Step 3: finalize submission
                        token = upload_info.get("token", "")
                        submit_resp = await client.post(
                            f"{KAGGLE_API}/competitions/submissions/submit/{comp}",
                            headers=headers,
                            json={"blobFileTokens": token, "submissionDescription": message},
                            timeout=30,
                        )
                        result = submit_resp.text
                    else:
                        result = f"Upload failed: {upload_resp.status_code}"
            else:
                # Fallback: try kaggle package
                raise RuntimeError(f"Direct upload init failed ({resp.status_code}), trying kaggle package")
    except Exception:
        # Fallback to kaggle Python package
        try:
            def _do_submit():
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                return api.competition_submit(
                    file_name=file_path,
                    message=message,
                    competition=comp,
                )
            result = await asyncio.to_thread(_do_submit)
        except Exception as e:
            return ToolResult(formatted=f"Submission failed: {e}", isError=True)

    # Track in local score history
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "competition": comp,
        "file": file_path,
        "message": message,
        "hypothesis": hypothesis,
        "score": "pending",  # Updated when score is available
        "api_result": str(result) if result else "ok",
    }
    _save_score(comp, entry)

    lines = [
        f"Submission to `{comp}` uploaded successfully.",
        f"**File**: {file_path}",
        f"**Message**: {message}",
    ]
    if hypothesis:
        lines.append(f"**Hypothesis**: {hypothesis}")
    lines.append(f"\nAPI response: {result}")
    lines.append("\nScore will appear in `my_submissions` once processed (usually 1-5 min).")
    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _push_notebook(args: dict, _limit: int) -> ToolResult:
    """Push a notebook to Kaggle and run it on their GPU/TPU.

    Reads a local .py or .ipynb file, wraps it as a kernel, and pushes
    it via the Kaggle kernels push API. The kernel runs remotely on
    Kaggle's infrastructure (GPU T4/P100/H100 as configured).
    """
    script_path = args.get("script_path", "") or args.get("file_path", "")
    competition = args.get("competition", "")
    title = args.get("title", "")
    gpu = args.get("gpu", True)
    internet = args.get("internet", True)
    accelerator = args.get("accelerator", "")  # e.g. "NvidiaRtxPro6000", "gpu" (P100)
    docker_image = args.get("docker_image", "")
    model_sources = args.get("model_sources", [])
    dataset_sources = args.get("dataset_sources", [])
    kernel_sources = args.get("kernel_sources", [])

    if not script_path:
        return ToolResult(formatted="Error: `script_path` is required (path to .py or .ipynb file).", isError=True)
    if not os.path.isfile(script_path):
        return ToolResult(formatted=f"Error: file not found: {script_path}", isError=True)

    headers = _require_auth()
    username = os.environ.get("KAGGLE_USERNAME", "agent")

    # Read the source file
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Determine kernel type
    is_notebook = script_path.endswith(".ipynb")
    kernel_type = "notebook" if is_notebook else "script"
    language = "python"

    # Generate a slug from the title or filename
    if not title:
        title = os.path.splitext(os.path.basename(script_path))[0].replace("_", "-")
        if competition:
            title = f"{competition[:30]}-{title[:15]}"
    slug = title.lower().replace(" ", "-")[:50]
    kernel_ref = f"{username}/{slug}"

    # For script type, we send the raw python source.
    # For notebook type, we send the JSON.
    if is_notebook:
        # Clean outputs from cells before pushing
        try:
            import nbformat
            nb = nbformat.reads(source, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
                if isinstance(cell.source, list):
                    cell.source = "".join(cell.source)
            script_body = nbformat.writes(nb)
        except Exception:
            script_body = source
    else:
        script_body = source

    # Build push request payload
    payload: dict[str, Any] = {
        "slug": kernel_ref,
        "newTitle": title,
        "text": script_body,
        "language": language,
        "kernelType": kernel_type,
        "isPrivate": True,
        "enableGpu": gpu,
        "enableInternet": internet,
        "datasetDataSources": dataset_sources,
        "competitionDataSources": [competition] if competition else [],
        "kernelDataSources": kernel_sources,
        "modelDataSources": model_sources,
        "categoryIds": [],
    }
    # Set machine shape / accelerator (e.g. "NvidiaRtxPro6000" for 48GB GPU)
    if accelerator:
        payload["machineShapeNullable"] = accelerator
    # Set custom docker image (competition-specific runtimes)
    if docker_image:
        payload["dockerImagePinningType"] = "original"
        payload["dockerImageNullable"] = docker_image

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{KAGGLE_API}/kernels/push",
            headers=headers,
            json=payload,
            timeout=60,
        )

    if resp.status_code not in (200, 201):
        return ToolResult(
            formatted=f"Kernel push failed ({resp.status_code}): {resp.text[:500]}",
            isError=True,
        )

    result = resp.json()
    version = result.get("versionNumber", "?")
    url = result.get("url", f"https://www.kaggle.com/code/{kernel_ref}")
    error = result.get("error", "")

    if error:
        return ToolResult(formatted=f"Kernel push error: {error}", isError=True)

    lines = [
        f"Kernel pushed successfully!",
        f"**Ref**: `{kernel_ref}`",
        f"**Version**: {version}",
        f"**GPU**: {'enabled' if gpu else 'disabled'}",
        f"**URL**: {url}",
        "",
        f"Check status with: `kaggle(operation=\"notebook_status\", notebook=\"{kernel_ref}\")`",
    ]
    if result.get("invalidDatasetSources"):
        lines.append(f"Warning — invalid dataset sources: {result['invalidDatasetSources']}")
    if result.get("invalidCompetitionSources"):
        lines.append(f"Warning — invalid competition sources: {result['invalidCompetitionSources']}")
    if result.get("invalidModelSources"):
        lines.append(f"Warning — invalid model sources: {result['invalidModelSources']}")

    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _notebook_status(args: dict, _limit: int) -> ToolResult:
    """Check the execution status of a pushed Kaggle notebook/kernel.

    Tries the kernels/status API first; if 403 (common with KGAT_ tokens),
    falls back to pulling kernel metadata which includes lastRunTime.
    """
    ref = args.get("notebook", "") or args.get("ref", "")
    if not ref:
        return ToolResult(formatted="Error: `notebook` parameter is required (e.g. 'user/slug').", isError=True)

    if "/" not in ref:
        username = os.environ.get("KAGGLE_USERNAME", "")
        ref = f"{username}/{ref}"

    parts = ref.split("/")
    headers = _require_auth()

    # Try the status endpoint first
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{KAGGLE_API}/kernels/status",
            headers=headers,
            params={"userName": parts[0], "kernelSlug": parts[-1]},
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unknown")
            failure_msg = data.get("failureMessage", "")

            emoji_map = {
                "queued": "[QUEUED]", "running": "[RUNNING]", "complete": "[COMPLETE]",
                "error": "[ERROR]", "cancelAcknowledged": "[CANCELLED]",
            }
            status_label = emoji_map.get(status, f"[{status.upper()}]")

            lines = [
                f"**Kernel status for `{ref}`**: {status_label} `{status}`",
            ]
            if failure_msg:
                lines.append(f"**Failure message**: {failure_msg}")
            if status == "complete":
                lines.append(f"\nGet output: `kaggle(operation=\"notebook_output\", notebook=\"{ref}\")`")
            elif status in ("queued", "running"):
                lines.append(f"\nStill running. Check again in a few minutes.")
            return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)

        # Fallback: try pulling kernel info (works with some token scopes)
        pull_resp = await client.get(
            f"{KAGGLE_API}/kernels/pull",
            headers=headers,
            params={"userName": parts[0], "kernelSlug": parts[-1]},
            timeout=15,
        )
        if pull_resp.status_code == 200:
            data = pull_resp.json()
            meta = data.get("metadata", {})
            last_run = meta.get("lastRunTime", "unknown")
            lines = [
                f"**Kernel `{ref}`** — pull succeeded",
                f"**Title**: {meta.get('title', '?')}",
                f"**Last run**: {last_run}",
                f"**GPU**: {meta.get('enableGpu', '?')}",
                "",
                f"Direct status API returned {resp.status_code} (token scope limitation).",
                f"Check notebook at: https://www.kaggle.com/code/{ref}",
            ]
            return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)

    # Neither worked
    url = f"https://www.kaggle.com/code/{ref}"
    lines = [
        f"**Kernel status API returned {resp.status_code}** for `{ref}`.",
        f"This is common with KGAT_ API tokens (limited scope).",
        f"",
        f"Check status manually: {url}",
        f"The kernel was pushed successfully — it should be running on Kaggle's infrastructure.",
    ]
    return ToolResult(formatted="\n".join(lines), totalResults=1, resultsShared=1)


async def _notebook_output(args: dict, _limit: int) -> ToolResult:
    """Download output files from a completed Kaggle kernel.

    Downloads the output to a local directory and lists the files.
    """
    ref = args.get("notebook", "") or args.get("ref", "")
    dest_dir = args.get("dest_dir", "")
    if not ref:
        return ToolResult(formatted="Error: `notebook` parameter is required (e.g. 'user/slug').", isError=True)

    if "/" not in ref:
        username = os.environ.get("KAGGLE_USERNAME", "")
        ref = f"{username}/{ref}"

    parts = ref.split("/")
    headers = _require_auth()

    if not dest_dir:
        dest_dir = os.path.join(os.getcwd(), "kaggle_output", parts[-1])
    os.makedirs(dest_dir, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(
            f"{KAGGLE_API}/kernels/output",
            headers=headers,
            params={"userName": parts[0], "kernelSlug": parts[-1]},
            timeout=120,
        )

    if resp.status_code != 200:
        return ToolResult(
            formatted=f"Output download failed ({resp.status_code}): {resp.text[:300]}",
            isError=True,
        )

    # The response may be a zip file or JSON with file list
    content_type = resp.headers.get("content-type", "")
    files_saved = []

    if "application/zip" in content_type or "application/octet-stream" in content_type:
        import zipfile
        import io
        zip_path = os.path.join(dest_dir, "output.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
            files_saved = zf.namelist()
    elif "application/json" in content_type:
        data = resp.json()
        if isinstance(data, dict) and "files" in data:
            for finfo in data["files"]:
                fname = finfo.get("fileName", "unknown")
                furl = finfo.get("url", "")
                if furl:
                    file_resp = await client.get(furl, headers=headers, timeout=120)
                    fpath = os.path.join(dest_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(file_resp.content)
                    files_saved.append(fname)
            if "log" in data:
                log_path = os.path.join(dest_dir, "kernel_log.txt")
                with open(log_path, "w") as f:
                    f.write(data["log"])
                files_saved.append("kernel_log.txt")
        else:
            # Save raw JSON response
            raw_path = os.path.join(dest_dir, "output.json")
            with open(raw_path, "w") as f:
                json.dump(data, f, indent=2)
            files_saved.append("output.json")
    else:
        # Save raw content
        raw_path = os.path.join(dest_dir, "output.bin")
        with open(raw_path, "wb") as f:
            f.write(resp.content)
        files_saved.append("output.bin")

    lines = [
        f"**Output for `{ref}`** saved to `{dest_dir}`",
        f"**Files** ({len(files_saved)}):",
    ]
    for fname in files_saved:
        fpath = os.path.join(dest_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            lines.append(f"  - `{fname}` ({size_kb:.1f} KB)")
        else:
            lines.append(f"  - `{fname}`")

    return ToolResult(formatted="\n".join(lines), totalResults=len(files_saved), resultsShared=len(files_saved))


async def _score_history(args: dict, _limit: int) -> ToolResult:
    """Show local score tracking history for a competition."""
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` parameter is required.", isError=True)

    scores = _load_scores(comp)
    if not scores:
        return ToolResult(formatted=f"No local score history for '{comp}'. Submit first.")

    lines = [f"**Score history for `{comp}`** ({len(scores)} submissions)\n"]

    numeric_scores = []
    for i, s in enumerate(scores, 1):
        score = s.get("score", "pending")
        lines.append(
            f"{i}. [{s.get('timestamp', '?')}] score: **{score}** "
            f"— {s.get('hypothesis', s.get('message', ''))}"
        )
        try:
            numeric_scores.append(float(score))
        except (ValueError, TypeError):
            pass

    if numeric_scores:
        lines.append(f"\n**Best**: {min(numeric_scores)}")
        lines.append(f"**Worst**: {max(numeric_scores)}")
        lines.append(f"**Average**: {sum(numeric_scores) / len(numeric_scores):.6f}")
        if len(numeric_scores) >= 2:
            trend = numeric_scores[-1] - numeric_scores[0]
            direction = "improving" if trend < 0 else "worsening" if trend > 0 else "flat"
            lines.append(f"**Trend**: {direction} ({trend:+.6f} from first to last)")

    return ToolResult(formatted="\n".join(lines), totalResults=len(scores), resultsShared=len(scores))


# ---------------------------------------------------------------------------
# Run log operations
# ---------------------------------------------------------------------------

async def _save_run_op(args: dict, _limit: int) -> ToolResult:
    """Save a run entry to the competition's run log.

    Use this to track every notebook push, error, fix, and submission.
    The agent should call this after every significant event.
    """
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` is required.", isError=True)

    entry = {
        "type": args.get("run_type", "notebook_push"),  # notebook_push, error, fix, submission, research
        "notebook": args.get("notebook", ""),
        "version": args.get("version", ""),
        "hypothesis": args.get("hypothesis", ""),
        "result": args.get("result", ""),  # success, error, oom, timeout, etc.
        "error_summary": args.get("error_summary", ""),
        "fix_applied": args.get("fix_applied", ""),
        "score": args.get("score", ""),
        "notes": args.get("notes", ""),
    }
    # Remove empty fields
    entry = {k: v for k, v in entry.items() if v}
    _save_run(comp, entry)

    count = len(_load_runs(comp))
    today_subs = _today_submission_count(comp)
    return ToolResult(
        formatted=f"Run logged (total: {count}, today's submissions: {today_subs}/3).",
        totalResults=1, resultsShared=1,
    )


async def _run_history(args: dict, limit: int) -> ToolResult:
    """View the full run history for a competition — errors, fixes, submissions, scores.

    The agent should read this at the START of every session to avoid repeating mistakes.
    """
    comp = args.get("competition", "")
    if not comp:
        return ToolResult(formatted="Error: `competition` is required.", isError=True)

    runs = _load_runs(comp)
    if not runs:
        return ToolResult(formatted=f"No run history for `{comp}`. This is a fresh start.", totalResults=0, resultsShared=0)

    today = time.strftime("%Y-%m-%d")
    today_subs = sum(1 for r in runs if r.get("type") == "submission" and r.get("timestamp", "").startswith(today))

    # Separate by type for summary
    errors = [r for r in runs if r.get("result") in ("error", "oom", "timeout")]
    submissions = [r for r in runs if r.get("type") == "submission"]
    best_score = None
    for s in submissions:
        try:
            sc = float(s.get("score", ""))
            if best_score is None or sc < best_score:
                best_score = sc
        except (ValueError, TypeError):
            pass

    lines = [
        f"# Run history for `{comp}`\n",
        f"**Total runs**: {len(runs)}",
        f"**Submissions**: {len(submissions)} (today: {today_subs}/3)",
        f"**Errors encountered**: {len(errors)}",
    ]
    if best_score is not None:
        lines.append(f"**Best score**: {best_score}")

    # Show errors and their fixes (critical for not repeating mistakes)
    if errors:
        lines.append("\n## Errors & fixes (DO NOT repeat these)")
        for e in errors:
            ts = e.get("timestamp", "?")
            err = e.get("error_summary", "unknown")
            fix = e.get("fix_applied", "none")
            lines.append(f"- [{ts}] **{err}** → fix: {fix}")

    # Show recent runs (last N)
    recent = runs[-limit:]
    lines.append(f"\n## Recent runs (last {len(recent)})")
    for r in recent:
        ts = r.get("timestamp", "?")
        rtype = r.get("type", "?")
        result = r.get("result", "?")
        hyp = r.get("hypothesis", "")
        score = r.get("score", "")
        notebook = r.get("notebook", "")
        parts = [f"[{ts}] **{rtype}**"]
        if notebook:
            parts.append(f"`{notebook}`")
        parts.append(f"→ {result}")
        if score:
            parts.append(f"(score: {score})")
        if hyp:
            parts.append(f"| hypothesis: {hyp}")
        lines.append("- " + " ".join(parts))

    return ToolResult(formatted="\n".join(lines), totalResults=len(runs), resultsShared=len(recent))


# ---------------------------------------------------------------------------
# Operation dispatch
# ---------------------------------------------------------------------------

_OPERATIONS: dict[str, Any] = {
    "list_competitions": _list_competitions,
    "competition_details": _competition_details,
    "list_data_files": _list_data_files,
    "list_notebooks": _list_notebooks,
    "read_notebook": _read_notebook,
    "notebook_metadata": _notebook_metadata,
    "list_discussions": _list_discussions,
    "read_discussion": _read_discussion,
    "leaderboard": _leaderboard,
    "my_submissions": _my_submissions,
    "submit": _submit,
    "push_notebook": _push_notebook,
    "notebook_status": _notebook_status,
    "notebook_output": _notebook_output,
    "score_history": _score_history,
    "save_run": _save_run_op,
    "run_history": _run_history,
}

# ---------------------------------------------------------------------------
# Tool spec + handler
# ---------------------------------------------------------------------------

KAGGLE_TOOL_SPEC: dict[str, Any] = {
    "name": "kaggle",
    "description": (
        "Kaggle competition tool — browse competitions, research notebooks and discussions, "
        "view leaderboards, submit predictions, run notebooks on Kaggle GPUs, and track scores.\n\n"
        "Operations:\n"
        "- list_competitions: List active competitions (optional: search, category, sort_by)\n"
        "- competition_details: Get full details for a competition (competition)\n"
        "- list_data_files: List competition data files (competition)\n"
        "- list_notebooks: List top notebooks/kernels (competition, sort_by: scoreAscending|voteCount|dateRun)\n"
        "- read_notebook: Read full notebook source as markdown (notebook: 'owner/slug')\n"
        "- notebook_metadata: Get notebook's data/model/kernel sources + GPU settings — use to copy exact sources before push_notebook\n"
        "- list_discussions: List forum discussions (competition)\n"
        "- read_discussion: Read a discussion topic (topic_id)\n"
        "- leaderboard: View top leaderboard entries (competition)\n"
        "- my_submissions: List your submissions + local tracking (competition)\n"
        "- submit: Submit predictions file (competition, file_path, message, hypothesis) — REQUIRES APPROVAL\n"
        "- push_notebook: Push a .py/.ipynb script to Kaggle and run it on GPU (script_path, competition, gpu, model_sources) — REQUIRES APPROVAL\n"
        "- notebook_status: Check execution status of a pushed kernel (notebook: 'user/slug')\n"
        "- notebook_output: Download output files from a completed kernel (notebook, dest_dir)\n"
        "- score_history: View local score tracking with trend analysis (competition)\n"
        "- save_run: Log a run event — notebook push, error, fix, submission (competition, run_type, hypothesis, result, error_summary, fix_applied, score)\n"
        "- run_history: View full run log with errors/fixes/scores — READ THIS AT SESSION START to avoid repeating mistakes (competition)\n\n"
        "Examples:\n"
        '  kaggle(operation="list_competitions", search="nlp")\n'
        '  kaggle(operation="list_notebooks", competition="titanic", sort_by="voteCount")\n'
        '  kaggle(operation="read_notebook", notebook="owner/kernel-slug")\n'
        '  kaggle(operation="push_notebook", script_path="./train.py", competition="comp-slug", gpu=true, '
        'model_sources=["metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"])\n'
        '  kaggle(operation="notebook_status", notebook="user/kernel-slug")\n'
        '  kaggle(operation="notebook_output", notebook="user/kernel-slug")\n'
        '  kaggle(operation="submit", competition="titanic", file_path="/path/to/submission.csv", '
        'message="XGBoost v2", hypothesis="Adding feature interactions improves AUC")\n'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform.",
                "enum": list(_OPERATIONS.keys()),
            },
            "competition": {
                "type": "string",
                "description": "Competition slug (e.g. 'titanic', 'nvidia-nemotron-reasoning-challenge').",
            },
            "search": {
                "type": "string",
                "description": "Search query for list_competitions.",
            },
            "category": {
                "type": "string",
                "description": "Filter competitions by category (e.g. 'featured', 'research', 'playground').",
            },
            "sort_by": {
                "type": "string",
                "description": "Sort order. For competitions: 'latestDeadline', 'prize'. For notebooks: 'scoreAscending', 'voteCount', 'dateRun'.",
            },
            "notebook": {
                "type": "string",
                "description": "Notebook ref as 'owner/slug' for read_notebook.",
            },
            "topic_id": {
                "type": "string",
                "description": "Discussion topic ID for read_discussion.",
            },
            "file_path": {
                "type": "string",
                "description": "Path to submission file for submit.",
            },
            "script_path": {
                "type": "string",
                "description": "Path to .py or .ipynb file for push_notebook.",
            },
            "gpu": {
                "type": "boolean",
                "description": "Enable GPU for push_notebook (default true).",
            },
            "internet": {
                "type": "boolean",
                "description": "Enable internet for push_notebook (default true).",
            },
            "accelerator": {
                "type": "string",
                "description": "GPU accelerator for push_notebook. Use notebook_metadata to find the right value. Common: 'NvidiaRtxPro6000' (48GB), 'gpu' (P100 16GB).",
            },
            "docker_image": {
                "type": "string",
                "description": "Custom docker image for push_notebook. Use notebook_metadata to get competition-specific images.",
            },
            "model_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Model sources for push_notebook (e.g. ['metric/nemotron-3-nano-30b-a3b-bf16/transformers/default']).",
            },
            "dataset_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dataset sources to attach to the kernel.",
            },
            "kernel_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Other kernel sources to attach (utility scripts).",
            },
            "title": {
                "type": "string",
                "description": "Kernel title for push_notebook.",
            },
            "dest_dir": {
                "type": "string",
                "description": "Local directory to save notebook_output files.",
            },
            "message": {
                "type": "string",
                "description": "Submission description message for submit.",
            },
            "hypothesis": {
                "type": "string",
                "description": "What you expect this submission to improve and why — tracked locally for learning.",
            },
            "run_type": {
                "type": "string",
                "description": "Type of run event for save_run: notebook_push, error, fix, submission, research.",
            },
            "result": {
                "type": "string",
                "description": "Run result for save_run: success, error, oom, timeout, etc.",
            },
            "error_summary": {
                "type": "string",
                "description": "Brief error description for save_run (e.g. 'ModuleNotFoundError: cutlass').",
            },
            "fix_applied": {
                "type": "string",
                "description": "What fix was applied for save_run (e.g. 'Added site.addsitedir for cutlass path').",
            },
            "version": {
                "type": "string",
                "description": "Notebook version for save_run tracking.",
            },
            "score": {
                "type": "string",
                "description": "Competition score for save_run submissions.",
            },
            "notes": {
                "type": "string",
                "description": "Free-form notes for save_run.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10, max 50).",
            },
        },
        "required": ["operation"],
    },
}


async def kaggle_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router."""
    operation = arguments.get("operation", "")
    handler = _OPERATIONS.get(operation)
    if not handler:
        ops = ", ".join(_OPERATIONS.keys())
        return f"Unknown operation '{operation}'. Valid: {ops}", False

    limit = min(arguments.get("limit", DEFAULT_LIMIT), MAX_LIMIT)

    try:
        result = await handler(arguments, limit)
        return result["formatted"], not result.get("isError", False)
    except ValueError as e:
        return str(e), False
    except Exception as e:
        logger.exception("Kaggle tool error: %s", e)
        return f"Kaggle tool error: {e}", False
