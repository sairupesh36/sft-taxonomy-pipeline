"""Read-only: apply DataTrove/FineWeb PIIFormatter regexes (copied verbatim) to real SFT rows; see what they would touch."""
import glob, ipaddress, json, os, random, re, collections
random.seed(3)
EMAIL = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[A-Za-z0-9](?:["
    r"A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|["
    r"01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])")
IP = re.compile(r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")
def is_public(s):
    try: return ipaddress.ip_address(s).is_global
    except ValueError: return False
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
files = [f for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")]
pick = random.sample(files, 60)
n = 0; rows_email = rows_ip_pub = rows_ip_priv = 0; ip_ex = []; em_ex = []; fake_email = 0
FAKE = re.compile(r"@(example\.(com|org|net)|test\.com|domain\.com|email\.com|mail\.com|yourdomain\.com|company\.com)\b", re.I)
for p in pick:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
        for _ in range(3000):
            line = f.readline()
            if not line: break
            n += 1
            d = json.loads(line)
            txt = "\n".join(m["content"] for m in d["messages"] if isinstance(m.get("content"), str))
            e = EMAIL.findall(txt) if "@" in txt else []
            if e:
                real = [x for x in EMAIL.finditer(txt) if not FAKE.search(x.group(0))]
                if real:
                    rows_email += 1
                    if len(em_ex) < 200: em_ex.append(txt[max(0, real[0].start()-40): real[0].end()+20].replace("\n", " "))
                else: fake_email += 1
            pub = priv = False
            for m in IP.finditer(txt):
                if is_public(m.group(0)):
                    pub = True
                    if len(ip_ex) < 400: ip_ex.append(txt[max(0, m.start()-45): m.end()+25].replace("\n", " "))
                else: priv = True
            rows_ip_pub += pub; rows_ip_priv += (priv and not pub)
print(f"rows {n:,}")
print(f"rows with a REAL-looking email (not example.com etc.): {rows_email:,} ({rows_email/n:.2%}); only placeholder emails: {fake_email:,}")
print(f"rows with a PUBLIC ip: {rows_ip_pub:,} ({rows_ip_pub/n:.2%}); only private/local ips: {rows_ip_priv:,}")
random.shuffle(em_ex); random.shuffle(ip_ex)
print("\n-- EMAIL contexts (16 random)");  [print("  ", repr(x)) for x in em_ex[:16]]
print("\n-- PUBLIC IP contexts (24 random)"); [print("  ", repr(x)) for x in ip_ex[:24]]
