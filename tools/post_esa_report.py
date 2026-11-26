import argparse
import json
import os
from contextlib import closing
from http.client import HTTPSConnection, HTTPException
from pathlib import Path


TEAM = "vdslab"
CATEGORY = "ゼミ/大学院/dan"
TOKEN_NAME = "ESA_ACCESS_TOKEN"
LOCAL_CONFIG = Path(__file__).resolve().parents[1] / ".env.esa.local"


def load_token():
    token = os.environ.get(TOKEN_NAME, "").strip()
    if token:
        return token

    try:
        lines = LOCAL_CONFIG.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ""

    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == TOKEN_NAME:
            return value.strip()
    return ""


def post_report(title, markdown, token):
    body = json.dumps(
        {"post": {
            "name": title.replace("#", "&#35;").replace("/", "&#47;"),
            "body_md": markdown,
            "category": CATEGORY,
            "wip": False,
        }},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    with closing(HTTPSConnection("api.esa.io", timeout=30)) as connection:
        connection.request("POST", f"/v1/teams/{TEAM}/posts", body, headers)
        response = connection.getresponse()
        if response.status != 201:
            raise RuntimeError(f"esa returned HTTP {response.status}.")
        url = json.load(response)["url"]

    if not isinstance(url, str) or not url:
        raise ValueError("esa response has no article URL")
    return url


def main():
    parser = argparse.ArgumentParser(description="Post a Markdown experiment report to esa.")
    parser.add_argument("--check-config", action="store_true", help="Check token presence offline.")
    parser.add_argument("--title", help="Title of the new esa article.")
    parser.add_argument("--report", type=Path, help="Path to the UTF-8 Markdown report.")
    args = parser.parse_args()

    if not args.check_config and (not args.title or not args.title.strip() or args.report is None):
        parser.error("--title and --report are required to post an article.")

    try:
        token = load_token()
    except (OSError, UnicodeError):
        parser.exit(1, "Could not read .env.esa.local as UTF-8.\n")
    if not token:
        parser.exit(1, "Set ESA_ACCESS_TOKEN in .env.esa.local or the environment.\n")
    if args.check_config:
        print("ESA_ACCESS_TOKEN is set. No network request was sent.")
        return

    try:
        markdown = args.report.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        parser.exit(1, "Could not read the report as UTF-8.\n")
    if not markdown.strip():
        parser.error("The Markdown report is empty.")

    try:
        url = post_report(args.title.strip(), markdown, token)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    except (OSError, HTTPException, ValueError, TypeError, KeyError):
        parser.exit(1, "Could not confirm the esa post. Check the category before retrying.\n")
    print(url)


if __name__ == "__main__":
    main()
