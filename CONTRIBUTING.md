# Contributing

Install the package in editable mode with the `dev` extra and the appropriate
compiled dependencies; see [installation](doc/installation.md). Use snake_case
module names and relative internal imports. Add regression coverage for behavior
changes and document new public APIs.

## Naming conventions

Names describe logical data structure independently of implementation type.

- Use singular nouns in folder, file, class, and function names. Folder examples
  include `doc/`, `example/`, `test/`, `transform/`, and `utility/`.
- Geometry nouns such as `node`, `edge`, `face`, `element`, `point`, and `vertex`
  may mean one or many entities. Keep them singular without automatically adding
  `_list` or `_table`. Preserve existing geometry identifiers and attributes,
  including `edge_length`, even when they hold whole collections.
- Use `index_list` for a generic index sequence and `xxx_index_list` when the
  entity is known, such as `node_index_list`, `edge_index_list`, `face_index_list`,
  or `element_index_list`. This applies to Python lists and tuples, NumPy arrays,
  and PyTorch tensors, including when the concrete type is explicitly specified.
- Other collection qualifiers describe logical meaning, not implementation.
  `_list` and `_table` are optional when they add useful meaning; storage shape
  alone does not require either suffix. A two-dimensional edge or point buffer
  can still be named `edge` or `point`.
- Use `xxx_set` when the logical collection requires unique entries and order has
  no meaning, independently of its storage implementation.
- Do not choose suffixes such as `_array`, `_tensor`, or `_tuple` merely because
  of the implementation type. Describe concrete types, dtypes, and devices in
  annotations or documentation.
- Keep names unchanged whether a collection contains zero, one, or many entries,
  and when its implementation changes without changing its logical structure.
- Preserve names defined by external packages and protocols, including VTK API
  names and pytest hooks such as `pytest_collection_modifyitems`. Do not rename
  an external function or class to satisfy this project convention.

For example, a known one-dimensional NumPy array of edge indices is still an
`edge_index_list`. Geometric edge and point collections remain `edge` and `point`
even if their storage has shape E x 2 or N x 3.

| Logical structure | Name |
| --- | --- |
| Sequence of indices | `index_list` |
| Sequence of edge indices | `edge_index_list` |
| One or many edge lengths | `edge_length` |
| One or many edges, each with two endpoint indices | `edge` |
| One or many points, each with three coordinates | `point` |
| Unordered collection of unique node indices | `node_index_set` |

Function names follow the same convention: `_source_edge` and `_face_edge`
describe geometry; `_index_list` describes an index sequence.

## Testing

Add tests under `test/` for new behavior and select the applicable registered
markers. Use synthetic data and temporary output paths whenever possible.
The remesher regression suite belongs under `test/remeshing/`.

Run the relevant tests before broader checks; see [testing](doc/testing.md) for
full and partial commands. Report test results and skipped dependencies explicitly.
Never commit local `data/`, generated outputs, or virtual environments.
