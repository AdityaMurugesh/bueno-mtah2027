"""Smart Tagging XML -> list of dicts with video-frame intervals (start, end inclusive)."""

import os
import xml.etree.ElementTree as ET

FPS = 25


def load(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "repo", "data",
                            "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter("instance"):
        rec = {"code": (inst.findtext("code") or "").strip().upper(),
               "start": int(round(float(inst.findtext("start")) * FPS)),
               "end": int(round(float(inst.findtext("end")) * FPS)), "tags": {}}
        for lab in inst.iter("label"):
            group = (lab.findtext("group") or "").strip().lower()
            rec["tags"][group] = (lab.findtext("text") or "").strip()
        out.append(rec)
    return out


if __name__ == "__main__":
    import collections
    ev = load()
    print(len(ev), "instances")
    print(collections.Counter(e["code"] for e in ev).most_common())
    print(collections.Counter(tuple(sorted(e["tags"])) for e in ev).most_common(6))
    print(collections.Counter(e["tags"].get("team") for e in ev))
