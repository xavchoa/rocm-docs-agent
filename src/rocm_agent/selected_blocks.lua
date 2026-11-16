-- Pandoc filter used by clean-docs.
-- The ROCm install pages use a website selector: `.. selected:: os=ubuntu os=debian` blocks
-- are shown only when the reader picks those options. Pandoc keeps them as Divs whose first
-- paragraph is the bare condition ("os=ubuntu os=debian"), so the condition reads like text.
-- This turns each block's condition into a visible bold label, the way AMD's own LLM export
-- (llms-full.txt) does, so chunkers that drop HTML comments (Docling) keep it:
--
--   **Applies to: install_method=pkgman; os=ubuntu,debian**
--   ...content...
--
-- Convention: a label applies until the next label or heading. The label carries the FULL
-- condition (this block's and every enclosing block's), so a reader never has to resolve
-- nesting. When content follows a nested block inside its parent, the parent's label is
-- printed again ("all" at page level). Same key twice: the inner (narrower) one wins. Values
-- of one key are alternatives (OR), different keys must all hold (AND), as on the site.

local NAMES = { i = "install_method", w = "use_case", fam = "gpu_family", gpu = "gpu" }
local ORDER = { "use_case", "install_method", "os", "gpu_family", "gfx", "gpu" }

local function parse(text)
  local cond, keys = {}, {}
  for token in text:gmatch("%S+") do
    local k, v = token:match("^([%w_-]+)=(.+)$")
    if k then
      k = NAMES[k] or k
      if not cond[k] then cond[k] = {}; table.insert(keys, k) end
      table.insert(cond[k], v)
    end
  end
  return cond, keys
end

local function rank(k)
  for i, name in ipairs(ORDER) do if name == k then return i end end
  return #ORDER + 1
end

local function render(cond)
  local keys = {}
  for k in pairs(cond) do table.insert(keys, k) end
  table.sort(keys, function(a, b)
    if rank(a) ~= rank(b) then return rank(a) < rank(b) end
    return a < b
  end)
  local parts = {}
  for _, k in ipairs(keys) do
    table.insert(parts, k .. "=" .. table.concat(cond[k], ","))
  end
  return table.concat(parts, "; ")
end

local function is_selected(b)
  return b.t == "Div" and b.classes:includes("selected")
end

-- The selector widget itself (`.. selector::` with its `.. selector-option::` buttons) is page
-- UI: it prints as loose words ("Installation method", "apt", "pip") and says nothing a
-- `selected` block's label does not already say. Dropped.
local function is_widget(b)
  return b.t == "Div" and (b.classes:includes("selector") or b.classes:includes("selector-option"))
end

local SEP = { pandoc.Str(";"), pandoc.Space() }

local function has_class(b, class)
  return b.t == "Div" and b.classes:includes(class)
end

local function join(parts)
  local out = pandoc.Inlines({})
  for i, p in ipairs(parts) do
    if i > 1 then out:extend(SEP) end
    out:extend(p)
  end
  return out
end

-- `.. matrix::` is the site's own table directive (rows of `.. matrix-cell::`). Pandoc does not
-- know it and prints one cell per line, which loses the rows. Rebuilt here as a real table.
-- A cell shown only for some selector choice (`:show-cond:`, or `.. selected::` inside it) keeps
-- its condition in brackets after the value; alternatives for one column share one cell.
local function cell_inlines(cell)
  local parts = {}
  for _, b in ipairs(cell.content) do
    if is_selected(b) then
      local first, body, own = b.content[1], b.content, ""
      if first and (first.t == "Para" or first.t == "Plain") then
        own = render((parse(pandoc.utils.stringify(first))))
        body = pandoc.Blocks({ table.unpack(b.content, 2) })
      end
      local inl = pandoc.utils.blocks_to_inlines(body, SEP)
      if own ~= "" then inl:extend({ pandoc.Space(), pandoc.Str("(" .. own .. ")") }) end
      table.insert(parts, inl)
    else
      table.insert(parts, pandoc.utils.blocks_to_inlines({ b }, SEP))
    end
  end
  local out = join(parts)
  local cond = cell.attributes["show-cond"] or ""
  if cond ~= "" then
    out:extend({ pandoc.Space(), pandoc.Str("(" .. render((parse(cond))) .. ")") })
  end
  return out
end

-- Rows sit directly in the matrix or inside a `.. matrix-head::` wrapper (header rows).
local function matrix_rows(div, found, in_head)
  for _, b in ipairs(div.content) do
    if has_class(b, "matrix-head") then
      matrix_rows(b, found, true)
    elseif has_class(b, "matrix-row") then
      table.insert(found, { row = b, head = in_head or b.attributes["header"] ~= nil })
    end
  end
  return found
end

-- The cells of one row as columns: { inl = Inlines, span = rowspan }. With `merge`, neighbouring
-- cells whose conditions differ are alternatives for ONE column (the site shows only one of
-- them) and share a cell. Neighbours with the same condition are separate columns.
local function row_columns(row, merge)
  local cols, prev = {}, nil
  for _, cell in ipairs(row.content) do
    if has_class(cell, "matrix-cell") then
      local cond = cell.attributes["show-cond"] or ""
      local span = tonumber(cell.attributes["rowspan"]) or 1
      if merge and cond ~= "" and prev and prev ~= "" and cond ~= prev then
        local last = cols[#cols]
        last.inl = join({ last.inl, cell_inlines(cell) })
        last.span = math.max(last.span, span)
      else
        table.insert(cols, { inl = cell_inlines(cell), span = span })
      end
      prev = cond
    end
  end
  return cols
end

local function matrix_table(div)
  local header, rows, ncols = nil, {}, 0
  local carry = {} -- column -> { inl, left }: a rowspan cell still covering later rows
  for _, item in ipairs(matrix_rows(div, {}, false)) do
    local row, active = item.row, 0
    for _, c in pairs(carry) do if c.left > 0 then active = active + 1 end end
    local cols = row_columns(row, true)
    if header and #cols + active ~= #header then -- merging broke the column count: undo it
      local plain = row_columns(row, false)
      if #plain + active == #header then cols = plain end
    end
    local cells, next_col, col = {}, 1, 1
    while next_col <= #cols or (carry[col] and carry[col].left > 0) do
      local c = carry[col]
      if c and c.left > 0 then -- copy the spanning cell into this row
        cells[col] = c.inl:clone()
        c.left = c.left - 1
      else
        local own = cols[next_col]
        next_col = next_col + 1
        cells[col] = own.inl
        if own.span > 1 then carry[col] = { inl = own.inl:clone(), left = own.span - 1 } end
      end
      col = col + 1
    end
    local row_cond = row.attributes["show-cond"] or ""
    if row_cond ~= "" and #cells > 0 then -- a whole row shown only for some choice
      cells[1]:extend({ pandoc.Space(), pandoc.Str("(" .. render((parse(row_cond))) .. ")") })
    end
    ncols = math.max(ncols, #cells)
    if item.head and not header then
      header = cells
    else
      table.insert(rows, cells)
    end
  end
  if ncols == 0 then return nil end
  local function padded(cells)
    local out = {}
    for i = 1, ncols do out[i] = { pandoc.Plain(cells and cells[i] or pandoc.Inlines({})) } end
    return out
  end
  local aligns, widths, body = {}, {}, {}
  for i = 1, ncols do aligns[i] = pandoc.AlignDefault; widths[i] = 0 end
  for _, cells in ipairs(rows) do table.insert(body, padded(cells)) end
  local simple = pandoc.SimpleTable({}, aligns, widths, padded(header), body)
  return pandoc.utils.from_simple_table(simple)
end

local function label(cond)
  local text = render(cond)
  if text == "" then text = "all" end
  return pandoc.Para({ pandoc.Strong(pandoc.Inlines("Applies to: " .. text)) })
end

local function convert(blocks, outer)
  local out = pandoc.Blocks({})
  local after_selected = false
  for _, b in ipairs(blocks) do
    if is_widget(b) then goto continue end
    if not is_selected(b) and after_selected and b.t ~= "Header" then
      out:insert(label(outer)) -- back to the enclosing block's condition
    end
    after_selected = is_selected(b)
    if is_selected(b) then
      local first = b.content[1]
      local own = {}
      local body = b.content
      if first and (first.t == "Para" or first.t == "Plain") then
        own = parse(pandoc.utils.stringify(first))
        body = pandoc.Blocks({ table.unpack(b.content, 2) })
      end
      local cond = {}
      for k, v in pairs(outer) do cond[k] = v end
      for k, v in pairs(own) do cond[k] = v end
      local heading = b.attributes["heading"]
      if heading and heading ~= "" then -- heading first: a heading ends the previous label
        local level = tonumber(b.attributes["heading-level"]) or 2
        out:insert(pandoc.Header(math.min(level, 6), pandoc.Inlines(heading)))
      end
      out:insert(label(cond))
      out:extend(convert(body, cond))
    elseif has_class(b, "matrix") then
      out:insert(matrix_table(b) or b)
    elseif b.t == "Div" or b.t == "BlockQuote" then
      b.content = convert(b.content, outer)
      out:insert(b)
    elseif b.t == "BulletList" or b.t == "OrderedList" then
      for i, item in ipairs(b.content) do
        b.content[i] = convert(item, outer)
      end
      out:insert(b)
    else
      out:insert(b)
    end
    ::continue::
  end
  return out
end

local function is_label(b)
  return b and b.t == "Para" and #b.content == 1 and b.content[1].t == "Strong"
    and pandoc.utils.stringify(b.content[1]):match("^Applies to: ") ~= nil
end

-- A label directly followed by a heading or another label covers nothing: drop it.
local function prune(blocks)
  local out = pandoc.Blocks({})
  for i, b in ipairs(blocks) do
    local nxt = blocks[i + 1]
    local nxt_first = nxt and (nxt.t == "Div" or nxt.t == "BlockQuote") and nxt.content[1]
    if not (is_label(b) and nxt and (nxt.t == "Header" or is_label(nxt) or is_label(nxt_first))) then
      out:insert(b)
    end
  end
  return out
end

return {
  { Pandoc = function(doc) doc.blocks = convert(doc.blocks, {}); return doc end },
  { Blocks = prune },
}
