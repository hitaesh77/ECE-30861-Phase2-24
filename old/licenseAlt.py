import asyncio
import re
import time
import aiohttp
from typing import Optional
from typing import Tuple

licenseID_Map = {
    
    "apache-2.0": "Apache-2.0", "apache license": "Apache-2.0",
    "mit license": "MIT", "mit": "MIT",
    "bsd 3-clause": "BSD-3-Clause", "bsd 2-clause": "BSD-2-Clause",
    "lgpl 2.1": "LGPL-2.1-only", "lgpl-2.1": "LGPL-2.1-only",
    "lgpl 3.0": "LGPL-3.0-only", "lgpl-3.0": "LGPL-3.0-only",
    "gpl v3": "GPL-3.0-only", "gpl-3.0": "GPL-3.0-only",
    "agpl v3": "AGPL-3.0-only", "agpl-3.0": "AGPL-3.0-only",
    "mpl 2.0": "MPL-2.0", "mpl-2.0": "MPL-2.0",
    "unlicense": "Unlicense", "isc": "ISC",
    "cc-by-4.0": "CC-BY-4.0", "cc-by-sa-4.0": "CC-BY-SA-4.0",
}

compatibility = {
    
    "MIT" : 1.0,
    "BSD-2-Clause" : 1.0,
    "BSD-3-Clause" : 1.0,
    "ISC" : 1.0,
    "Unlicense" : 1.0,
    "Zlib" : 1.0,
    "LGPL-2.1-only" : 1.0,
    "LGPL-3.0-only" : 1.0,
    "Apache-2.0" : 0.8,
    "MPL-2.0" : 0.6,
    "GPL-2.0-only" : 0,
    "GPL-3.0-only" : 0,
    "AGPL-3.0-only" : 0,
    
   
}

def licenseIdExtraction(text: str) -> Optional(str):
    lowerText = text.lower()
    m = re.search(r"^license:\s*([A-Za-z0-9\.\-+]+)\s*$", text, flags = re.M|re.I)
    if m:
        key = m.group(1).lower()
        return licenseID_Map.get(key, key.upper())
    for key, lic in licenseID_Map.items():
        if key in lowerText:
            return lic
    return None


def clarityCheck(source: str, lic: Optional[str]) -> float:
    if source.upper().startswith("LICENSE") or source.upper().startswith("COPYING"):
        return 1.0
    if source.upper().startswith("README"):
        return 0.8
    if not lic:
        return 0
    return 0.6
    
def compatibilityCheck(lic: Optional[str]) -> float:
    if not lic:
        return 0
    return compatibility.get(lic, 0.2)
