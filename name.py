pat = re.compile(get_name_pattern(get_name_list()))

for m in pat.finditer(text):
    # 沒有身分證 → 跳過
    if not m.group('twid'):
        continue

    parts = [f"主名：{m.group('name')}"]

    if m.group('alias'):   # 有別名才加
        parts.append(f"原名：{m.group('alias')}")

    parts.append(f"身分證：{m.group('twid')}")

    print("；".join(parts))