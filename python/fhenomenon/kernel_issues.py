"""Prepare local kernel request drafts and explicitly publish them to GitHub.com.

Publication needs only the Python standard library; model capture is imported
only for the draft subcommand. Tokens are read from the environment by the CLI.
"""

import argparse
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


def _validate(draft, repository):
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]+", repository
    ) or repository.split("/")[-1] in {".", ".."}:
        raise ValueError("repository must be an explicit GitHub.com owner/repo")
    identity = draft.get("request_id")
    if type(draft.get("schema_version")) is not int or draft["schema_version"] != 1:
        raise ValueError("draft requires schema_version 1")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("draft requires a SHA-256 request_id")
    marker = f"<!-- fhenomenon-kernel-request:v1:{identity} -->"
    body, title = draft.get("body"), draft.get("title")
    if (
        not isinstance(body, str)
        or not body.startswith(marker + "\n")
        or len(body) > 60000
    ):
        raise ValueError(
            "draft body must begin with its request marker and fit 60000 characters"
        )
    if (
        not isinstance(title, str)
        or not title.strip()
        or len(title) > 240
        or re.search(r"[\x00-\x1f]", title)
    ):
        raise ValueError("draft requires a short single-line title")
    return marker


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a token to a location supplied by a redirect.
        return None


def publish_kernel_issue(draft, repository, token):
    """Create once or return the existing (even closed) issue without editing it.

    This uses GitHub's repository issue listing, not the delayed search index.
    Serialize concurrent publishers per repository (the supplied workflow does).
    GitHub has no atomic create-if-absent API: external concurrent callers can
    still race. No POST is retried after an ambiguous network failure.
    """
    marker = _validate(draft, repository)
    if not isinstance(token, str) or not token.strip() or re.search(r"\s", token):
        raise ValueError(
            "publication requires GH_TOKEN or GITHUB_TOKEN with Issues:write"
        )
    endpoint = f"https://api.github.com/repos/{repository}/issues"
    opener = build_opener(_NoRedirect())

    def request(method, suffix="", payload=None):
        call = Request(
            endpoint + suffix,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "fhenomenon-kernel-requests",
            },
            method=method,
        )
        try:
            with opener.open(call, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(
                f"GitHub {method} failed (HTTP {error.code}); no automatic retry"
            ) from None
        except (URLError, TimeoutError, OSError, ValueError):
            caution = (
                " POST may have succeeded; inspect existing issues before rerunning."
                if method == "POST"
                else " No issue was created."
            )
            raise RuntimeError(
                f"GitHub {method} response unavailable or invalid.{caution}"
            ) from None

    def result(item, status):
        number = item.get("number") if isinstance(item, dict) else None
        if type(number) is not int or number < 1:
            raise RuntimeError(
                "GitHub response lacks an issue number; inspect existing issues before rerunning"
            )
        return {
            "status": status,
            "number": number,
            "url": f"https://github.com/{repository}/issues/{number}",
            "request_id": draft["request_id"],
        }

    for page in range(1, 101):
        items = request(
            "GET", f"?state=all&per_page=100&page={page}&sort=created&direction=asc"
        )
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            raise RuntimeError("invalid GitHub issue list; no issue was created")
        for item in items:
            if (
                "pull_request" not in item
                and marker in (item.get("body") or "").splitlines()
            ):
                return result(item, "existing")
        if len(items) < 100:
            created = request(
                "POST", payload={"title": draft["title"], "body": draft["body"]}
            )
            return result(created, "created")
    raise RuntimeError("issue listing exceeded 100 pages; no issue was created")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "draft", help="derive local drafts from a measured report; no network"
    )
    prepare.add_argument("report", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--context", type=Path)
    prepare.add_argument("--limit", type=int, default=1)
    submit = commands.add_parser(
        "submit", help="inspect a draft or explicitly publish it"
    )
    submit.add_argument("draft", type=Path)
    submit.add_argument("--repository", required=True)
    submit.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "draft":
            from .ml.requests import derive_kernel_issues

            report = json.loads(args.report.read_text())
            if report.get("passed") is False:
                raise ValueError("refusing a report with failed numerical validation")
            context = (
                json.loads(args.context.read_text())
                if args.context
                else report.get("issue_context", {})
            )
            drafts = derive_kernel_issues(
                report["kernel_requests"],
                report.get("kernel_cost_measurement"),
                context=context,
                limit=args.limit,
            )
            args.output.mkdir(parents=True, exist_ok=True)
            for draft in drafts:
                stem = args.output / draft["request_id"]
                stem.with_suffix(".json").write_text(json.dumps(draft, indent=2) + "\n")
                stem.with_suffix(".md").write_text(draft["body"])
            result = {"draft_count": len(drafts), "output": str(args.output)}
        else:
            draft = json.loads(args.draft.read_text())
            _validate(draft, args.repository)
            if args.publish:
                result = publish_kernel_issue(
                    draft,
                    args.repository,
                    os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", ""),
                )
            else:
                result = {
                    "status": "dry_run",
                    "repository": args.repository,
                    "title": draft["title"],
                    "request_id": draft["request_id"],
                }
        print(json.dumps(result))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
