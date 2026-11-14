-- Pandoc filter used by clean-docs.
-- GitHub Markdown tables only hold one line of inline text per cell. Cells holding lists or
-- several paragraphs make pandoc give up and print "[TABLE]". This flattens each such cell
-- to one line ("item one; item two") so the table survives as a normal Markdown table.

local function flatten_cell(cell)
  local blocks = cell.contents
  if #blocks == 1 and (blocks[1].t == "Plain" or blocks[1].t == "Para") then
    cell.contents = { pandoc.Plain(blocks[1].content) }
    return cell
  end
  local parts = {}
  for _, b in ipairs(blocks) do
    if b.t == "BulletList" or b.t == "OrderedList" then
      for _, item in ipairs(b.content) do
        table.insert(parts, pandoc.utils.stringify(item))
      end
    else
      local s = pandoc.utils.stringify(b)
      if s ~= "" then table.insert(parts, s) end
    end
  end
  cell.contents = { pandoc.Plain({ pandoc.Str(table.concat(parts, "; ")) }) }
  return cell
end

local function flatten_rows(rows)
  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      row.cells[i] = flatten_cell(cell)
    end
  end
end

function Table(t)
  flatten_rows(t.head.rows)
  for _, body in ipairs(t.bodies) do
    flatten_rows(body.head)
    flatten_rows(body.body)
  end
  flatten_rows(t.foot.rows)
  -- column widths force pandoc to keep a grid layout; GitHub tables have none
  for i, spec in ipairs(t.colspecs) do
    t.colspecs[i] = { spec[1], pandoc.ColWidthDefault }
  end
  return t
end
