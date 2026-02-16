import re

from pydantic import BaseModel, Field


class PatchHunk(BaseModel):
    pre: list[str] = Field(default_factory=list)
    old: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    post: list[str] = Field(default_factory=list)

    def old_block(self) -> str:
        return "".join(self.pre + self.old + self.post)

    def new_block(self) -> str:
        return "".join(self.pre + self.new + self.post)


class Patch(BaseModel):
    hunks: list[PatchHunk] = Field(default_factory=list)
    raw_text: str = ""


HUNK_SPLIT_RE = re.compile(r"(?m)^@@\s*$")


def parse_patch(patch: str) -> Patch:
    """
    Parse hunks-only patch text into a Patch object.
    """
    if patch.strip() == "NO_CHANGES":
        return Patch(hunks=[], raw_text=patch)

    chunks = HUNK_SPLIT_RE.split(patch)
    hunks_raw = chunks[1:]

    hunks: list[PatchHunk] = []
    for raw in hunks_raw:
        pre: list[str] = []
        old: list[str] = []
        new: list[str] = []
        post: list[str] = []
        state = "pre"

        for line in raw.splitlines(keepends=True):
            if line.strip() == "":
                continue

            prefix = line[0]
            content = line[1:]

            if prefix == " ":
                if state == "pre":
                    pre.append(content)
                elif state in ("old", "new"):
                    state = "post"
                    post.append(content)
                else:
                    post.append(content)
            elif prefix == "-":
                if state == "pre":
                    state = "old"
                old.append(content)
            elif prefix == "+":
                state = "new"
                new.append(content)
            else:
                raise ValueError(f"Invalid patch line (bad prefix): {line!r}")

        if not old and not new:
            # empty hunk, skip
            continue
        hunks.append(PatchHunk(pre=pre, old=old, new=new, post=post))

    return Patch(hunks=hunks, raw_text=patch)


def apply_patch(file_content: str, patch: Patch) -> tuple[str, list[int]]:
    """
    Apply a parsed Patch to file_content.

    Returns updated content and list of failed hunk indices.
    """
    content = file_content
    failed: list[int] = []

    for idx, hunk in enumerate(patch.hunks):
        old_block = hunk.old_block()
        new_block = hunk.new_block()

        pos = content.find(old_block)
        if pos == -1:
            failed.append(idx)
            continue

        content = content[:pos] + new_block + content[pos + len(old_block) :]

    return content, failed
