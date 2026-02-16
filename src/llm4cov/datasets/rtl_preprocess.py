import re

from tree_sitter import Node

from llm4cov.datasets.types import DataContext


# Keep existing lightweight preprocessing.
# Tree-sitter can usually tolerate comments, but macros/backticks may create ERROR nodes.
def remove_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def remove_line_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def strip_macros(text: str) -> str:
    text = re.sub(r"^\s*`define\s+\w+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*`(ifdef|ifndef|elsif|endif|else)\b.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`\w+(\([^)]*\))?", "", text)
    return text


def preprocess_verilog(text: str) -> str:
    text = remove_block_comments(text)
    text = remove_line_comments(text)
    text = strip_macros(text)
    return text


def _iter_named(node: Node) -> list[Node]:
    # tree_sitter.Node has .named_children
    children = getattr(node, "named_children", None)
    if not children:
        return []
    return list(children)


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_first_identifier(node: Node, src: bytes) -> str | None:
    """
    Best-effort: find the first identifier-like descendant.
    This avoids hard-coding exact node field names.
    """
    IDENT_TYPES = {
        "identifier",
        "simple_identifier",
        "module_identifier",
        "interface_identifier",
        "class_identifier",
        "tf_identifier",
    }

    stack: list[Node] = [node]
    while stack:
        n = stack.pop()
        if n.type in IDENT_TYPES:
            return _node_text(n, src).strip()
        # prefer named children (skips punctuation tokens)
        for ch in reversed(_iter_named(n)):
            stack.append(ch)
    return None


def _extract_module_name(module_decl_node: Node, src: bytes) -> str | None:
    """
    Try common field names first, then fall back to identifier search.
    """
    for field in ("name", "identifier", "module_identifier"):
        try:
            c = module_decl_node.child_by_field_name(field)
        except Exception:
            c = None
        if c is not None:
            s = _node_text(c, src).strip()
            if s:
                return s
    return _find_first_identifier(module_decl_node, src)


def _extract_instantiated_module_name(inst_node: Node, src: bytes) -> str | None:
    """
    For a module instantiation statement, the first identifier-like token is
    typically the instantiated module type (before instance name).
    """
    # Some grammars expose the module type via a field; try a few
    for field in ("module", "type", "name", "module_type"):
        try:
            c = inst_node.child_by_field_name(field)
        except Exception:
            c = None
        if c is not None:
            s = _find_first_identifier(c, src) or _node_text(c, src).strip()
            if s:
                return s

    # Fallback: pick the first identifier-like descendant
    return _find_first_identifier(inst_node, src)


def extract_potential_top(code_files: list[str]) -> set[str]:
    """
    Returns modules that are defined but never instantiated (as a module type),
    restricted to the set of modules defined in the provided files.
    """
    import tree_sitter_systemverilog as tssv
    from tree_sitter import Language, Parser

    sv_lang = Language(tssv.language())
    parser = Parser(sv_lang)

    MODULE_DECL_TYPES = {"module_declaration"}
    INST_TYPES = {"module_instantiation"}

    defined: set[str] = set()
    instantiated: set[str] = set()

    for code in code_files:
        code = preprocess_verilog(code)
        src = code.encode("utf-8", errors="replace")
        tree = parser.parse(src)
        root = tree.root_node

        def walk(node: Node, current_module: str | None, source: bytes) -> None:
            nonlocal defined, instantiated

            # Enter module scope
            if node.type in MODULE_DECL_TYPES:
                m = _extract_module_name(node, source)
                if m:
                    defined.add(m)
                    current_module = m

            # Record instantiation only when inside a module (avoids picking up junk in errors)
            if current_module is not None and node.type in INST_TYPES:
                t = _extract_instantiated_module_name(node, source)
                if t:
                    instantiated.add(t)

            for ch in _iter_named(node):
                walk(ch, current_module, source)

        walk(root, None, src)

    if len(defined) == 1:
        return set(defined)

    # Only treat as instantiated if the type is also defined in this corpus.
    instantiated_defined = {m for m in instantiated if m in defined}
    return defined - instantiated_defined


def context_filter_top_extract(context: DataContext, debug: bool = False) -> str:
    rtl_code_files = [f.content for f in context.rtl_files]
    potential_tops = extract_potential_top(rtl_code_files)

    if len(potential_tops) == 0:
        raise ValueError("No potential top module found in the provided RTL files.")
    if len(potential_tops) > 1:
        if debug:
            print(f"Debug: potential tops found: {potential_tops}")
            print(f"problem id: {context.id}, dataset: {context.dataset_name}")
            for rtl_file in context.rtl_files:
                print(f"--- RTL file: {rtl_file.name} ---")
                print(rtl_file.content)
        raise ValueError(
            "Multiple potential top modules found, Please specify the top module explicitly."
        )
    return potential_tops.pop()
