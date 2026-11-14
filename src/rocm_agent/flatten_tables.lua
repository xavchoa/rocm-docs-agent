-- Pandoc filter used by clean-docs.
-- GitHub Markdown tables only hold one line of inline text per cell. Cells holding lists,
-- several paragraphs or line breaks make pandoc give up and print "[TABLE]". This flattens
-- each such cell to one line ("item one; item two") so the table survives as a normal
-- Markdown table. Cells spanning several rows (HTML rowspan, rst grid tables) are copied
-- into every row they cover, so each row reads on its own.

local function one_line(inlines)
  return pandoc.Inlines(inlines):walk({
    LineBreak = function() return pandoc.Space() end,
    SoftBreak = function() return pandoc.Space() end,
  })
end

-- rst line blocks ("| ``VAR``" / "| description") -> "`VAR`; description", formatting kept
local function join_lines(lines)
  local out = pandoc.Inlines({})
  for i, line in ipairs(lines) do
    if #line > 0 then
      if #out > 0 then out:extend({ pandoc.Str(";"), pandoc.Space() }) end
      out:extend(one_line(line))
    end
  end
  return out
end

local function flatten_cell(cell)
  local blocks = cell.contents
  if #blocks == 0 then return cell end
  if #blocks == 1 and (blocks[1].t == "Plain" or blocks[1].t == "Para") then
    cell.contents = { pandoc.Plain(one_line(blocks[1].content)) }
    return cell
  end
  if #blocks == 1 and blocks[1].t == "LineBlock" then
    cell.contents = { pandoc.Plain(join_lines(blocks[1].content)) }
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

local function copy_cell(cell)
  return pandoc.Cell(cell.contents, cell.alignment, 1, cell.col_span, cell.attr)
end

-- Replace each rowspan > 1 by copies of the cell in the rows below it.
local function expand_rowspans(rows)
  local pending = {} -- column index -> { cell = ..., left = rows still to fill }
  for _, row in ipairs(rows) do
    local out, col = {}, 1
    local function fill_pending()
      while pending[col] do
        local p = pending[col]
        table.insert(out, copy_cell(p.cell))
        p.left = p.left - 1
        if p.left == 0 then pending[col] = nil end
        col = col + p.cell.col_span
      end
    end
    for _, cell in ipairs(row.cells) do
      fill_pending()
      if cell.row_span > 1 then
        pending[col] = { cell = cell, left = cell.row_span - 1 }
        cell.row_span = 1
      end
      table.insert(out, cell)
      col = col + cell.col_span
    end
    fill_pending()
    row.cells = out
  end
end

local function flatten_rows(rows)
  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      row.cells[i] = flatten_cell(cell)
    end
  end
end

-- Sphinx cross-reference roles pandoc does not know (:doc:`HIPCC <hipcc:index>`) arrive as code
-- spans; keep only the link text, like the regex in clean.py does outside tables.
function Code(c)
  local role = c.attributes["role"]
  if c.classes:includes("interpreted-text") and (role == "doc" or role == "ref") then
    local text = c.text:match("^(.-)%s*<[^>]*>$") or c.text
    return pandoc.Str(text)
  end
end

function Table(t)
  flatten_rows(t.head.rows)
  for _, body in ipairs(t.bodies) do
    expand_rowspans(body.body)
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
