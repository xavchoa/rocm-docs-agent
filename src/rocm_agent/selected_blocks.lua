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

local function label(cond)
  local text = render(cond)
  if text == "" then text = "all" end
  return pandoc.Para({ pandoc.Strong(pandoc.Inlines("Applies to: " .. text)) })
end

local function convert(blocks, outer)
  local out = pandoc.Blocks({})
  local after_selected = false
  for _, b in ipairs(blocks) do
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
