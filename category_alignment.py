"""
Bridges the two category vocabularies used by TechPrime AI:

  MODEL categories  - what the XGBoost models are trained on (33 POS buckets: MEMORY, DISPLAY, COOLING,
                      LAPTOP PR2, ...). Keys of category_code_map.json. Internal only.
  SHOP vocabulary   - what the PHP system shows: 8 Client menu GROUPS (Component, Peripherals, ...) and their
                      LABELS (Memory, Monitor, CPU Cooling, ...). Mirrors ep_shop_taxonomy() in
                      TechPrime-AI/includes/client_shop_taxonomy.php - KEEP THE TWO IN SYNC.

Public API used by inventory_forecasting.py:
  classify_product(msku, product_name, model_category) -> (parent_key, shop_group, shop_label)
  resolve_filter(value) -> ('label'|'group'|'model', canonical_value)      # raises ValueError if unknown
  taxonomy()            -> {'groups': {group: [labels]}}
  group_of(label)       -> group name ('Others' if unknown)
  reload_map()          -> clears caches (call after editing the taxonomy / code map)

Filter strings accepted by resolve_filter (case-insensitive): a shop label ("Memory", "CPU Cooling"),
a shop group ("Component"), a few legacy bucket names ("Display", "RAM", "GPU", ...), or an explicit
internal model category with the "model:" prefix ("model:MEMORY"). Labels/groups win over model names.
"""
import functools
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
CATEGORY_MAP_PATH = BASE_DIR / 'category_code_map.json'

# group label -> (parent key, [labels])   -- same order as the Client menu
_TAXONOMY = {
    'Component': ('component', ['Chassis Fan', 'CPU Cooling', 'Graphics Card', 'Hard Disk', 'Memory', 'Motherboard',
                                'PC Case', 'Power Supply', 'Processor AMD', 'Processor INTEL', 'Processor Tray',
                                'Solid State Drive']),
    'Peripherals': ('peripherals', ['CCTV', 'Headset', 'Keyboard', 'Keyboard and Mouse', 'Monitor', 'Mouse',
                                    'Printer & Scanner', 'Projector', 'Recorder', 'Speaker', 'UPS & AVR',
                                    'Web & Digital Camera']),
    'Accessories': ('accessories', ['Cables', 'Earphones', 'Gaming Surface', 'Power Bank']),
    'PC Furnitures': ('pc-furnitures', ['Chairs', 'Tables']),
    'OS & Softwares': ('os-softwares', ['Antivirus', 'Office Applications', 'Operating System']),
    'Laptops And Mobile Devices': ('laptops-mobile', ['Chromebook', 'Laptops', 'Mobile Phone', 'Tablet']),
    'Desktop': ('desktop', ['Desktop']),
    'Others': ('others', ['Others']),
}

# Legacy product bucket / import names -> shop label (mirrors ias_category_legacy_aliases(), first alias)
_LEGACY = {
    'display': 'Monitor', 'audio': 'Headset', 'cooling': 'CPU Cooling', 'gpu': 'Graphics Card',
    'graphic card': 'Graphics Card', 'ram': 'Memory', 'psu': 'Power Supply', 'case': 'PC Case',
    'printers and scanners': 'Printer & Scanner', 'printer and scanner': 'Printer & Scanner',
    'cables and adapters': 'Cables', 'cameras': 'Web & Digital Camera', 'laptop': 'Laptops',
    'ssd': 'Solid State Drive', 'hdd': 'Hard Disk', 'ups': 'UPS & AVR', 'mobile': 'Mobile Phone',
}

# Default shop label for each model category (before name-based refinement)
_MODEL_DEFAULT = {
    'ACCESSORIES': 'Others', 'ALL': 'Others', 'AUDIO': 'Headset', 'CABLES AND ADAPTERS': 'Cables',
    'CAMERA': 'Web & Digital Camera', 'COMBO': 'Keyboard and Mouse', 'COOLING': 'CPU Cooling',
    'CUSTOMIZATION': 'Others', 'DISPLAY': 'Monitor', 'EXTERNAL STORAGE DEVICES': 'Hard Disk',
    'GAMING SURFACE': 'Gaming Surface', 'GRAPHIC CARD': 'Graphics Card', 'HARD DISK': 'Hard Disk',
    'KEYBOARD': 'Keyboard', 'MEMORY': 'Memory', 'MINI PC': 'Desktop', 'MOTHERBOARD': 'Motherboard',
    'MOUSE': 'Mouse', 'NETWORK DEVICE': 'Others', 'PC CASE': 'PC Case', 'POWER SUPPLY': 'Power Supply',
    'PRINTER AND SCANNER': 'Printer & Scanner', 'PROCESSOR': 'Processor INTEL',
    'SOLID STATE DRIVE': 'Solid State Drive', 'SPEAKER': 'Speaker', 'UPS & AVR': 'UPS & AVR',
}

# (regex on the lower-cased product name, label) - first match wins; only consulted for the listed model categories
_REFINE = {
    'COOLING': [(r'\b(chassis fan|case fan|sickleflow|argb fan|fan freebie)\b', 'Chassis Fan'),
                (r'\bfan\b(?!.*(cpu|cooler|heatsink|liquid|aio))', 'Chassis Fan')],
    'PROCESSOR': [(r'\b(tray|ttp)\b', 'Processor Tray'),
                  (r'\b(ryzen|athlon|amd)\b', 'Processor AMD'),
                  (r'\b(intel|core|pentium|celeron|xeon)\b', 'Processor INTEL')],
    'AUDIO': [(r'\b(earphones?|earbuds?|in-ear|iem)\b', 'Earphones'),
              (r'\bheadsets?\b', 'Headset'),
              (r'\b(speakers?|subwoofer|soundbar)\b', 'Speaker')],
    'CAMERA': [(r'\b(nvr|dvr|recorder)\b', 'Recorder'),
               (r'\b(cctv|ip camera|dome|bullet|hiwatch|hikvision|ezviz)\b', 'CCTV')],
    'DISPLAY': [(r'\bprojector\b', 'Projector')],
    'EXTERNAL STORAGE DEVICES': [(r'\b(ssd|solid state)\b', 'Solid State Drive')],
    'ACCESSORIES': [(r'\bpower ?bank\b', 'Power Bank'),
                    (r'\b(cable|adapter|converter|hdmi|vga|displayport|usb hub)\b', 'Cables'),
                    (r'\b(earphones?|earbuds?)\b', 'Earphones'),
                    (r'\b(mouse ?pad|gaming surface)\b', 'Gaming Surface'),
                    (r'\b(chair)\b', 'Chairs'), (r'\b(table|desk)\b', 'Tables')],
    'MINI PC': [(r'\bchromebook\b', 'Chromebook')],
    'CUSTOMIZATION': [],
}
_POWER_STATION = re.compile(r'\b(power station|portable power station|bluetti)\b')


@functools.lru_cache(maxsize=1)
def _label_index():
    """lower-cased label -> (parent_key, group, label); and group lower -> (parent_key, group)."""
    labels, groups = {}, {}
    for group, (key, subs) in _TAXONOMY.items():
        groups[group.lower()] = (key, group)
        for lab in subs:
            labels.setdefault(lab.lower(), (key, group, lab))
    return labels, groups


@functools.lru_cache(maxsize=1)
def _model_categories():
    with open(CATEGORY_MAP_PATH) as f:
        return {k.upper(): k for k in json.load(f)}          # upper -> exact key (e.g. 'ALL' -> 'All')


def reload_map():
    _label_index.cache_clear()
    _model_categories.cache_clear()


def taxonomy():
    return {'groups': {g: list(subs) for g, (_, subs) in _TAXONOMY.items()}}


def group_of(label):
    hit = _label_index()[0].get(str(label).strip().lower())
    return hit[1] if hit else 'Others'


def _place(label):
    return _label_index()[0][label.lower()]


def classify_product(msku, product_name, model_category):
    """Return (parent_key, shop_group, shop_label) for one product. Name-based refinement first, then the model
    category default. Laptops (LAPTOP GA1/PR2/...) all map to 'Laptops'. Never raises; unknown -> Others."""
    name = str(product_name or '').lower()
    mc = str(model_category or '').strip().upper()
    if _POWER_STATION.search(name) and not re.search(r'\b(solar panel|foldable solar)\b', name):
        return _place('Others')
    if mc.startswith('LAPTOP'):
        return _place('Chromebook' if re.search(r'\bchromebook\b', name) else 'Laptops')
    for pattern, label in _REFINE.get(mc, []):
        if re.search(pattern, name):
            return _place(label)
    return _place(_MODEL_DEFAULT.get(mc, 'Others'))


def resolve_filter(value):
    """Turn a user-supplied category into ('label'|'group'|'model', canonical_value)."""
    v = str(value or '').strip()
    if not v:
        raise ValueError('Empty category.')
    labels, groups = _label_index()
    models = _model_categories()
    low = v.lower()
    if low.startswith('model:'):
        key = models.get(v[6:].strip().upper())
        if key is None:
            raise ValueError(f"Unknown model category '{v[6:].strip()}'.")
        return 'model', key
    if low in labels:
        return 'label', labels[low][2]
    if low in groups:
        return 'group', groups[low][1]
    if low in _LEGACY:
        return 'label', _LEGACY[low]
    if v.upper() in models:                                  # bare internal name, e.g. 'LAPTOP PR2'
        return 'model', models[v.upper()]
    raise ValueError(f"Unknown category '{v}'. Use a shop label, a shop group, or 'model:<NAME>'.")