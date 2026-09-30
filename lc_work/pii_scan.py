import os, re, sys, json, random, collections
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise"
FAKE_DOM = re.compile(r"(?:^|\.)(?:example|test|domain|email|mail|yourdomain|yourcompany|company|sample|foo|bar|placeholder|localhost|invalid|acme|mydomain|website|site|host)\.(?:com|org|net|edu|io|co|local|invalid)$|^(?:user|name|you|your|username|email|john|jane|someone|info|admin)@", re.I)
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}(?![\w-])")
PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?|\(\d{3}\)\s?\d{3}[\s.-]\d{4}|\d{3}[.-]\d{3}[.-]\d{4})(?![\w.-]*\d)")
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
SECRET = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
URLCRED = re.compile(r"[a-z]{3,10}://[^\s:/@]{1,40}:[^\s:/@]{1,40}@[\w.-]+")
def luhn(d):
    d = [int(c) for c in d][::-1]; return sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2]) % 10 == 0 if False else (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0
PRIVATE_IP = re.compile(r"^(?:10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|0\.|255\.|169\.254\.)")
def scan_text(t):
    out = {}
    e = [m.group(0) for m in EMAIL.finditer(t)]; e_real = [x for x in e if not FAKE_DOM.search(x)]
    if e: out["email_any"] = e
    if e_real: out["email_real_looking"] = e_real
    p = PHONE.findall(t)
    if p: out["phone"] = p
    ips = [x for x in IPV4.findall(t) if not PRIVATE_IP.match(x) and len({int(o) for o in x.split('.')}) > 1 and not x.startswith(("1.1.", "1.0.", "2.0."))]
    if ips: out["ipv4_public_looking"] = ips
    c = []
    for m in CARD.finditer(t):
        d = re.sub(r"\D", "", m.group(0))
        if 13 <= len(d) <= 19 and len(set(d)) > 3 and luhn(d) and re.search(r"[ -]", m.group(0)) : c.append(m.group(0))
    if c: out["credit_card_luhn_spaced"] = c
    s = SSN.findall(t)
    if s: out["us_ssn_pattern"] = s
    k = SECRET.findall(t)
    if k: out["secret_or_key"] = k
    u = URLCRED.findall(t)
    if u: out["url_with_password"] = u
    return out
def worker(args):
    lang, fn, n, seed = args; rnd = random.Random(seed); p = f"{ROOT}/{lang}/{fn}"; size = os.path.getsize(p); cnt = collections.Counter(); ex = collections.defaultdict(list); seen = set(); rows = 0
    with open(p, "rb") as f:
        for _ in range(n * 3):
            if rows >= n: break
            f.seek(rnd.randint(0, max(0, size - 1))); f.readline(); off = f.tell(); l = f.readline()
            if not l or off in seen: continue
            seen.add(off)
            try: m = json.loads(l)["messages"]
            except Exception: continue
            rows += 1; text = "\n".join(x["content"] for x in m if isinstance(x.get("content"), str))
            for k, v in scan_text(text).items():
                cnt[k] += 1
                if len(ex[k]) < 4:
                    i = text.find(v[0]); ex[k].append(re.sub(r"\s+", " ", text[max(0, i - 35): i + len(v[0]) + 25]))
    return lang, rows, cnt, ex
if __name__ == "__main__":
    tasks = []; i = 0
    for fn in sorted(os.listdir(f"{ROOT}/en")): tasks.append(("en", fn, 1500, 100 + i)); i += 1
    for lang in ("hi", "zh", "es", "ar", "vi", "ta", "kn", "bn"):
        fs = sorted(os.listdir(f"{ROOT}/{lang}"))[:6]
        for fn in fs: tasks.append((lang, fn, 400, 900 + i)); i += 1
    with Pool(24) as pool: res = pool.map(worker, tasks)
    groups = {"en": collections.Counter(), "other": collections.Counter()}; rows = {"en": 0, "other": 0}; ex = collections.defaultdict(list)
    for lang, r, c, e in res:
        g = "en" if lang == "en" else "other"; rows[g] += r; groups[g].update(c)
        for k, v in e.items(): ex[k] += v[: max(0, 4 - len(ex[k]))]
    print(f"rows scanned: en {rows['en']:,} | other languages {rows['other']:,}")
    print(f"{'pattern':28s}{'en rows':>10s}{'en %':>8s}{'other %':>9s}")
    for k in sorted(set(groups['en']) | set(groups['other']), key=lambda k: -groups['en'][k]):
        print(f"{k:28s}{groups['en'][k]:>10,}{100*groups['en'][k]/rows['en']:>8.2f}{100*groups['other'][k]/max(1,rows['other']):>9.2f}")
    json.dump(ex, open("lc_work/pii_examples.json", "w"))
    print("\nEXAMPLES (context around the match):")
    for k, v in ex.items():
        print(f"[{k}]"); [print("   ", repr(x[:130])) for x in v[:3]]
