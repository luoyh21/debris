"""验证分段策略能否带来新记录"""
import requests, json, time

def fetch(query, limit=1000, retry=3):
    for i in range(retry):
        try:
            r = requests.get("http://www.asterank.com/api/asterank",
                params={"query": json.dumps(query), "limit": limit},
                timeout=30, headers={"User-Agent": "space_debris/1.0"})
            r.raise_for_status()
            return r.json() or []
        except Exception as e:
            print(f"  重试 {i+1}/{retry}: {e}")
            time.sleep(3)
    return []

# 默认 MBA
d = fetch({"class": "MBA"})
default_names = {x.get("full_name") for x in d}
print(f"default MBA: {len(default_names)} 条")
time.sleep(2)

# 分段 a < 2.5 (内主带)
s1 = fetch({"class": "MBA", "a": {"$lt": 2.5}})
s1_names = {x.get("full_name") for x in s1}
new1 = s1_names - default_names
print(f"a<2.5: {len(s1_names)} 条，新增: {len(new1)}")
time.sleep(2)

# 分段 a >= 3.2 (外主带)
s2 = fetch({"class": "MBA", "a": {"$gte": 3.2}})
s2_names = {x.get("full_name") for x in s2}
new2 = s2_names - default_names
print(f"a>=3.2: {len(s2_names)} 条，新增: {len(new2)}")

print(f"\n合计可新增: {len(new1 | new2)} 条")
