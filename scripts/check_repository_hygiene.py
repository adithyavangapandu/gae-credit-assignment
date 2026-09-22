"""Fail when generated data, figures, reports, or website files are tracked."""

import subprocess

FORBIDDEN_PREFIXES = ("data/", "reports/", "research_site/")
FORBIDDEN_SUFFIXES = (
    ".csv",
    ".feather",
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".svg",
)


def main() -> int:
    tracked = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8").splitlines()
    forbidden = [
        path
        for path in tracked
        if path.startswith(FORBIDDEN_PREFIXES) or path.lower().endswith(FORBIDDEN_SUFFIXES)
    ]
    if forbidden:
        print("Generated data, figures, reports, or website files are tracked:")
        for path in forbidden:
            print(f"  {path}")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
