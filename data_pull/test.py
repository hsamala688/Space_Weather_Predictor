import requests

url = "https://cddis.nasa.gov/archive/gnss/products/ionex/2010/001/codg0010.10i.Z"

s = requests.Session()
s.trust_env = True  # read ~/.netrc

r = s.get(url, allow_redirects=True, timeout=60)

print("final status:", r.status_code)
print("final URL:   ", r.url)
print("redirect chain:")
for h in r.history:
    print("   ", h.status_code, h.url)
print("first 200 bytes:")
print(r.content[:200])