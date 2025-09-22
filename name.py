pat = re.compile(get_name_pattern(get_name_list()))

for m in pat.finditer(text):
    parts = [f"主名：{m.group('name')}"]

    if m.group('alias'):   # 只有有別名才加
        parts.append(f"原名：{m.group('alias')}")

    if m.group('twid'):    # 只有有身分證才加
        parts.append(f"身分證：{m.group('twid')}")

    print("；".join(parts))