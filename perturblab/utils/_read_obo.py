from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from perturblab.types.math import DAG

from .logging import get_logger

logger = get_logger()


def read_obo(obo_path: str, load_obsolete: bool = False) -> Tuple[List[Dict], "DAG"]:
    """Parse OBO file and return GO term information and hierarchy.

    This is a standalone parser that extracts GO term metadata and their
    hierarchical relationships (is_a) from an OBO file.

    Args:
        obo_path: Path to the OBO file.
        load_obsolete: Whether to include obsolete terms.

    Returns:
        Tuple containing:
        - List[Dict]: List of GO term dictionaries with keys:
            - 'id': GO term ID (e.g., 'GO:0006915')
            - 'name': Human-readable name
            - 'namespace': One of 'biological_process', 'molecular_function', 'cellular_component'
            - 'definition': Detailed definition
            - 'is_obsolete': Boolean flag
            - 'alt_ids': Set of alternative IDs
        - DAG: Directed acyclic graph of GO term relationships (parent -> child edges)

    Raises:
        FileNotFoundError: If OBO file doesn't exist.

    Example:
        >>> terms, dag = read_obo('go-basic.obo')
        >>> print(f"Loaded {len(terms)} GO terms")
        >>> print(f"DAG has {dag.n_edges} edges")
        >>>
        >>> # Access term info
        >>> term_dict = {t['id']: t for t in terms}
        >>> print(term_dict['GO:0006915']['name'])
        >>>
        >>> # Query hierarchy
        >>> parents = dag.get_parents('GO:0006915')
    """
    if not Path(obo_path).exists():
        raise FileNotFoundError(
            f"OBO file not found: {obo_path}\n"
            "Download from: http://current.geneontology.org/ontology/go-basic.obo"
        )

    logger.info(f"Parsing OBO file: {obo_path}")

    terms = []
    edges = []
    format_version = None
    data_version = None
    default_namespace = "default"

    with open(obo_path, "r") as f:
        in_header = True
        current_term = None

        for line in f:
            line = line.strip()

            # Parse header
            if in_header:
                if line.startswith("format-version:"):
                    format_version = line.split(":", 1)[1].strip()
                elif line.startswith("data-version:"):
                    data_version = line.split(":", 1)[1].strip()
                elif line.startswith("default-namespace:"):
                    default_namespace = line.split(":", 1)[1].strip()
                elif line == "[Term]":
                    in_header = False
                    current_term = {
                        "id": "",
                        "name": "",
                        "namespace": default_namespace,
                        "definition": "",
                        "is_obsolete": False,
                        "alt_ids": set(),
                        "_parent_ids": set(),  # Temporary storage for parsing
                    }
                continue

            # Start of new term
            if line == "[Term]":
                if current_term is not None:
                    # Skip obsolete terms if requested
                    if not current_term["is_obsolete"] or load_obsolete:
                        terms.append(current_term)
                        # Add edges from this term to its parents
                        for parent_id in current_term["_parent_ids"]:
                            edges.append((parent_id, current_term["id"]))

                current_term = {
                    "id": "",
                    "name": "",
                    "namespace": default_namespace,
                    "definition": "",
                    "is_obsolete": False,
                    "alt_ids": set(),
                    "_parent_ids": set(),
                }

            # Skip typedef sections
            elif line == "[Typedef]":
                if current_term is not None:
                    if not current_term["is_obsolete"] or load_obsolete:
                        terms.append(current_term)
                        for parent_id in current_term["_parent_ids"]:
                            edges.append((parent_id, current_term["id"]))
                    current_term = None

            # Parse term fields
            elif current_term is not None and ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()

                if key == "id":
                    current_term["id"] = value
                elif key == "name":
                    current_term["name"] = value
                elif key == "namespace":
                    current_term["namespace"] = value
                elif key == "def":
                    # Definition is in quotes
                    if value.startswith('"'):
                        end_quote = value.find('"', 1)
                        if end_quote != -1:
                            current_term["definition"] = value[1:end_quote]
                elif key == "is_a":
                    # is_a line format: "GO:0008150 ! biological_process"
                    parent_id = value.split()[0]
                    current_term["_parent_ids"].add(parent_id)
                elif key == "alt_id":
                    current_term["alt_ids"].add(value)
                elif key == "is_obsolete" and value == "true":
                    current_term["is_obsolete"] = True

        # Add last term
        if current_term is not None:
            if not current_term["is_obsolete"] or load_obsolete:
                terms.append(current_term)
                for parent_id in current_term["_parent_ids"]:
                    edges.append((parent_id, current_term["id"]))

    # Remove temporary field
    for term in terms:
        del term["_parent_ids"]

    # Import DAG here to avoid circular import
    from perturblab.types.math import DAG

    dag = DAG(edges, validate=False) if edges else DAG([], validate=False)

    logger.info(
        f"Parsed {len(terms)} GO terms, {len(edges)} edges "
        f"(format: {format_version}, version: {data_version})"
    )

    return terms, dag
