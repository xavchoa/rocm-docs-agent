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
    md = _clean_rst(page, "10.0.0")
    assert "[TABLE]" not in md
    assert "| Tarball | Linux; Windows |" in md
