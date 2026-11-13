"""Publication protocol against a fake GitHub endpoint, without network or tokens."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fhenomenon.kernel_issues import main, publish_kernel_issue


def draft():
    identity = "a" * 64
    return {
        "schema_version": 1,
        "request_id": identity,
        "title": "Kernel request: aten.mm.default",
        "body": f"<!-- fhenomenon-kernel-request:v1:{identity} -->\nReviewed contract",
    }


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())


class Endpoint:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Response(response)


class GithubIssueTests(unittest.TestCase):
    def publish(self, endpoint, value=None, repo="owner/repo"):
        with patch("fhenomenon.kernel_issues.build_opener", return_value=endpoint):
            return publish_kernel_issue(value or draft(), repo, "test-secret-token")

    def test_create_submits_only_title_body_and_scoped_token(self):
        api = Endpoint(
            [], {"number": 4, "html_url": "https://github.com/owner/repo/issues/4"}
        )
        result = self.publish(api)
        self.assertEqual(result["status"], "created")
        self.assertEqual([r.get_method() for r in api.calls], ["GET", "POST"])
        self.assertEqual(
            json.loads(api.calls[-1].data), {k: draft()[k] for k in ("title", "body")}
        )
        self.assertEqual(
            api.calls[-1].get_header("Authorization"), "Bearer test-secret-token"
        )

    def test_closed_issue_is_reused_without_update_or_new_notification(self):
        api = Endpoint(
            [
                {
                    "number": 2,
                    "state": "closed",
                    "body": draft()["body"],
                    "html_url": "https://github.com/owner/repo/issues/2",
                }
            ]
        )
        result = self.publish(api)
        self.assertEqual(result["status"], "existing")
        self.assertEqual(len(api.calls), 1)
        self.assertIn("state=all", api.calls[0].full_url)

    def test_paginates_and_ignores_pull_requests(self):
        first = [{"number": i, "body": "unrelated"} for i in range(99)]
        first.append({"number": 100, "body": draft()["body"], "pull_request": {}})
        api = Endpoint(
            first,
            [
                {
                    "number": 102,
                    "body": draft()["body"],
                    "html_url": "https://github.com/owner/repo/issues/102",
                }
            ],
        )
        self.assertEqual(self.publish(api)["status"], "existing")
        self.assertIn("page=2", api.calls[-1].full_url)
        self.assertEqual(len(api.calls), 2)

    def test_failed_search_does_not_create_issue_or_leak_error_response(self):
        for error in (
            HTTPError("https://api.github.com", 403, "test-secret-token", {}, None),
            URLError("test-secret-token"),
        ):
            api = Endpoint(error)
            with self.assertRaises(RuntimeError) as caught:
                self.publish(api)
            self.assertNotIn("test-secret-token", str(caught.exception))
            self.assertEqual(len(api.calls), 1)

    def test_failed_post_is_not_blindly_retried(self):
        api = Endpoint([], URLError("connection lost"))
        with self.assertRaisesRegex(RuntimeError, "may have succeeded"):
            self.publish(api)
        self.assertEqual(len(api.calls), 2)

    def test_invalid_repository_marker_or_missing_token_never_connects(self):
        api = Endpoint()
        for repo in (
            "owner/repo/issues",
            "../repo",
            "https://github.com/owner/repo",
            "owner/repo?x",
        ):
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                self.publish(api, repo=repo)
        invalid = draft()
        invalid["body"] = "unreviewable"
        with self.assertRaises(ValueError):
            self.publish(api, invalid)
        with self.assertRaises(ValueError):
            publish_kernel_issue(draft(), "owner/repo", "")
        self.assertEqual(api.calls, [])

    def test_cli_submit_is_offline_until_publish_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.json"
            path.write_text(json.dumps(draft()))
            with (
                patch("fhenomenon.kernel_issues.publish_kernel_issue") as publish,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(
                    main(["submit", str(path), "--repository", "owner/repo"]), 0
                )
                publish.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())["status"], "dry_run")

    def test_cli_drafts_from_full_report_without_copying_training_values(self):
        from test_kernel_requests import request_artifact
        from test_costs import measurement

        request = request_artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            out = Path(directory) / "drafts"
            path.write_text(
                json.dumps(
                    {
                        "kernel_requests": request,
                        "kernel_cost_measurement": measurement(request),
                        "training_values": "PRIVATE_VALUE",
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["draft", str(path), "--output", str(out)]), 0)
            files = list(out.glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertNotIn("PRIVATE_VALUE", files[0].read_text())
            self.assertTrue(files[0].with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
