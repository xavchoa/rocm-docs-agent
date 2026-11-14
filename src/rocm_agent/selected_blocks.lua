-- Pandoc filter used by clean-docs.
-- The ROCm install pages use a website selector: `.. selected:: os=ubuntu os=debian` blocks
-- are shown only when the reader picks those options. Pandoc keeps them as Divs whose first
-- paragraph is the bare condition ("os=ubuntu os=debian"), so the condition reads like text.
-- This replaces each block with explicit markers:
--
--   <!-- applies: install_method=pkgman; os=ubuntu,debian -->
--   ...content...
--   <!-- /applies -->
--
-- The opening marker carries the FULL condition (this block's and every enclosing block's),
-- so a reader never has to resolve nesting to know what a line applies to; the closing marker
-- returns to the enclosing block's condition. Same key twice: the inner (narrower) one wins.
-- Values of one key are alternatives (OR), different keys must all hold (AND), as on the site.

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

local function convert(blocks, outer)
  local out = pandoc.Blocks({})
  for _, b in ipairs(blocks) do
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
      out:insert(pandoc.RawBlock("markdown", "<!-- applies: " .. render(cond) .. " -->"))
      local heading = b.attributes["heading"]
      if heading and heading ~= "" then
        local level = tonumber(b.attributes["heading-level"]) or 2
        out:insert(pandoc.Header(math.min(level, 6), pandoc.Inlines(heading)))
      end
      out:extend(convert(body, cond))
      out:insert(pandoc.RawBlock("markdown", "<!-- /applies -->"))
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

function Pandoc(doc)
  doc.blocks = convert(doc.blocks, {})
  return doc
end
