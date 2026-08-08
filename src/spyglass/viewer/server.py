"""Local HTTP server for the spyglass results viewer.

Serves a small static frontend plus a JSON index built from a results
folder (the output of a `spyglass model.gguf` run), so the browser can
scrub through per-layer weight heatmaps the way a radiology viewer scrubs
through slices.
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from spyglass.viewer.index import build_index

STATIC_DIR = Path(__file__).parent / "static"


class ViewerRequestHandler(BaseHTTPRequestHandler):
    results_dir: Path

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        sys.stderr.write(f"[spyglass-view] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            path = "/index.html"

        try:
            if path == "/api/index":
                self._serve_json(build_index(self.results_dir))
            elif path.startswith("/data/"):
                query = parse_qs(parsed.query)
                transpose = query.get("transpose", ["0"])[0] == "1"
                self._serve_data_file(path[len("/data/") :], transpose=transpose)
            else:
                self._serve_static_file(path.lstrip("/"))
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:  # keep the server alive on unexpected errors
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_data_file(self, name: str, *, transpose: bool = False) -> None:
        # results_dir is flat (spyglass never writes subdirectories into it), so
        # reject anything that isn't a bare filename to keep this off the filesystem
        # outside results_dir.
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise FileNotFoundError(name)
        file_path = self.results_dir / name
        if transpose and file_path.suffix.lower() == ".png":
            self._send_transposed_png(file_path)
        else:
            self._send_file(file_path, cache=True)

    def _send_transposed_png(self, file_path: Path) -> None:
        # A true transpose (swap x/y, no mirroring) rather than a rotation, so a
        # channel's index keeps counting in the same direction it does in every
        # un-transposed image -- the viewer uses this to line every tensor's
        # residual-stream axis up as the horizontal one for visual comparison.
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        with Image.open(file_path) as img:
            body_io = io.BytesIO()
            img.transpose(Image.Transpose.TRANSPOSE).save(body_io, format="PNG")
        body = body_io.getvalue()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static_file(self, name: str) -> None:
        candidate = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            raise FileNotFoundError(name)
        self._send_file(candidate, cache=False)

    def _send_file(self, file_path: Path, *, cache: bool) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(
    results_dir: Path, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False
) -> None:
    ViewerRequestHandler.results_dir = results_dir
    server = ThreadingHTTPServer((host, port), ViewerRequestHandler)
    url = f"http://{host}:{server.server_port}/"
    print(f"spyglass viewer serving {results_dir} at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open_new_tab(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spyglass-view",
        description="Serve an MRI-style layer viewer for a spyglass results folder.",
    )
    parser.add_argument("results_dir", type=Path, help="a directory written by `spyglass`")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser", action="store_true", help="don't auto-open a browser tab"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results_dir: Path = args.results_dir

    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    serve(results_dir, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
