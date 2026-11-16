from rocm_agent.clean import _clean_myst, _tidy, _title


def test_myst_cleanup():
    raw = (
        "(my-label)=\n# ROCm {{ ROCM_VERSION }} notes\n\n"
        "```{note}\nUse the new installer.\n```\n\n"
        'See {doc}`the guide </install/rocm>` and <span class="x">AMD&trade;</span>.\n'
    )
    out = _tidy(_clean_myst(raw, "10.0.0"))
    assert "# ROCm 10.0.0 notes" in out
    assert "(my-label)=" not in out
    assert "> **Note:**" in out and "> Use the new installer." in out
    assert "See the guide and AMD™." in out


def test_title_fallback():
    assert _title("no heading here", "page") == "page"
    assert _title("# Hello\n", "page") == "Hello"


def test_list_table_with_bullet_cells_survives(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(
        ".. list-table::\n   :header-rows: 1\n   :widths: 30 70\n\n"
        "   * - Method\n     - Platform\n"
        "   * - Tarball\n     - - Linux\n       - Windows\n",
        encoding="utf-8",
    )
    md, _ = _clean_rst(page, "10.0.0")
    assert "[TABLE]" not in md
    assert "| Tarball | Linux; Windows |" in md


def test_selected_blocks_become_visible_applies_labels(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(
        ".. selected:: i=pkgman\n\n"
        "   .. selected:: os=ubuntu os=debian\n\n"
        "      Ubuntu step.\n\n"
        "   Any-OS step.\n\n"
        "Page step.\n",
        encoding="utf-8",
    )
    md, _ = _clean_rst(page, "10.0.0")
    assert "os=ubuntu os=debian" not in md  # no bare condition lines
    lines = [ln for ln in md.splitlines() if ln.strip()]
    assert lines == [  # the outer block's own label covers nothing here, so it is dropped
        "**Applies to: install_method=pkgman; os=ubuntu,debian**",
        "Ubuntu step.",
        "**Applies to: install_method=pkgman**",  # back to the parent's condition
        "Any-OS step.",
        "**Applies to: all**",  # back to page level
        "Page step.",
    ]


def test_selected_block_heading_option_becomes_heading(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(
        ".. selected:: i=runfile\n   :heading: Download the runfile\n   :heading-level: 3\n\n"
        "   Text.\n",
        encoding="utf-8",
    )
    md, _ = _clean_rst(page, "10.0.0")
    assert "### Download the runfile\n\n**Applies to: install_method=runfile**" in _tidy(md)


def test_html_table_in_myst_with_rowspan_and_breaks():
    raw = (
        "## T\n\n<table>\n<thead><tr><th>Group</th><th>Item</th></tr></thead>\n<tbody>\n"
        '<tr><td rowspan="2"><strong>Core</strong></td><td>a<br><br>b</td></tr>\n'
        "<tr><td>c</td></tr>\n</tbody>\n</table>\n"
    )
    out = _tidy(_clean_myst(raw, "10.0.0"))
    assert "<t" not in out and "[TABLE]" not in out
    rows = [" ".join(line.split()) for line in out.splitlines()]  # ignore column padding
    assert "| **Core** | a b |" in rows
    assert "| **Core** | c |" in rows  # rowspan copied into the next row


def test_myst_grid_cards_and_style_removed():
    raw = (
        "<style>\n.x { color: red; }\n</style>\n\n# Home\n\n"
        "::::{grid} 1 1 2 2\n:gutter: 3\n\n:::{grid-item-card} Core SDK\nThe libraries.\n"
        "+++\n[Go](x)\n:::\n::::\n"
    )
    out = _tidy(_clean_myst(raw, "10.0.0"))
    assert out.startswith("# Home")
    assert "{grid" not in out and ":::" not in out and "+++" not in out
    assert "**Core SDK**\n\nThe libraries." in out


def test_remote_content_is_spliced_and_label_before_table_parses(tmp_path):
    from rocm_agent.clean import _clean_rst

    remote = tmp_path / "_remote" / "ROCm" / "repo" / "docs" / "env.rst"
    remote.parent.mkdir(parents=True)
    remote.write_text(
        "SKIPPED TITLE\n\n.. _env:\n.. list-table::\n   :header-rows: 1\n\n"
        "   * - Variable\n     - Value\n"
        "   * - | ``HIP_PLATFORM``\n       | The platform.\n     - ``amd``\n",
        encoding="utf-8",
    )
    page = tmp_path / "p.rst"
    page.write_text(
        "Intro.\n\n.. remote-content::\n   :repo: ROCm/repo\n   :path: /docs/env.rst\n"
        "   :default_branch: develop\n   :start_line: 2\n\n"
        ".. remote-content::\n   :repo: ROCm/repo\n   :path: docs/gone.rst\n",
        encoding="utf-8",
    )
    md, missing = _clean_rst(page, "10.0.0", tmp_path)
    assert "SKIPPED TITLE" not in md
    assert "| `HIP_PLATFORM`; The platform. | `amd` |" in md
    assert missing == ["ROCm/repo/docs/gone.rst"]


def test_bare_line_block_spacers_and_script_dropped(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(
        "Last step.\n\n|\n|\n|\n\n.. raw:: html\n\n   <script>\n     alert(1);\n   </script>\n",
        encoding="utf-8",
    )
    md, _ = _clean_rst(page, "10.0.0")
    out = _tidy(md)
    assert out == "Last step.\n"


def test_selector_widget_is_dropped(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(
        ".. selector::\n   :key: os\n\n"
        "   .. selector-option:: Ubuntu\n      :value: ubuntu\n\n"
        "   .. selector-option:: RHEL\n      :value: rhel\n\n"
        "After the widget.\n",
        encoding="utf-8",
    )
    md, _ = _clean_rst(page, "10.0.0")
    assert _tidy(md) == "After the widget.\n"


MATRIX_RST = """.. matrix::

   .. matrix-head::

      .. matrix-row::

         .. matrix-cell::

            Name

         .. matrix-cell::

            Value

   .. matrix-row::

      .. matrix-cell::
         :rowspan: 2

         Group

      .. matrix-cell::
         :show-cond: os=ubuntu

         apt

      .. matrix-cell::
         :show-cond: os=rhel

         dnf

   .. matrix-row::
      :show-cond: os=sles

      .. matrix-cell::

         zypper
"""


def test_matrix_becomes_a_pipe_table(tmp_path):
    from rocm_agent.clean import _clean_rst

    page = tmp_path / "p.rst"
    page.write_text(MATRIX_RST, encoding="utf-8")
    md, _ = _clean_rst(page, "10.0.0")
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in md.splitlines() if ln.strip()]
    assert rows[0] == ["Name", "Value"]  # matrix-head is the header row
    assert set(rows[1][0]) <= {"-"}  # the separator line
    # neighbouring cells with different show-cond share one column, condition in brackets
    assert rows[2] == ["Group", "apt (os=ubuntu); dnf (os=rhel)"]
    # rowspan copied into the next row; the row-level show-cond lands on its first cell
    assert rows[3] == ["Group (os=sles)", "zypper"]
