import re

def get_name_pattern(name_list):
    # 姓氏 alternation（先 escape，避免特殊字元）
    surname_alt = r"(?:%s)" % "|".join(map(re.escape, name_list))
    # 主名：單姓 + 1~2 個中文字（→ 二字或三字姓名）
    name_core   = rf"(?P<name>{surname_alt}[\u4E00-\u9FFF]{{1,2}})"

    # 別名（支援：即 XXX、（原名/又名/曾用名：XXX））→ 整段可選
    alias_full  = rf"{surname_alt}[\u4E00-\u9FFF]{{1,2}}"
    alias_block = (
        rf"(?:"
        rf"(?:(?:即\s*)|（(?:原名|又名|曾用名)[:：]?\s*)(?P<alias>{alias_full})(?:）)?"
        rf")?"
    )

    # 身分證（A+9碼），可選；中間允許 0–20 任意字元
    twid_block  = r"(?:[\s\S]{0,20}(?P<twid>[A-Z][0-9]{9}))?"

    return name_core + alias_block + twid_block

pat = re.compile(get_name_pattern(get_name_list()))
for m in pat.finditer(text):
    print(
        f"主名：{m.group('name')}；"
        f"別名：{m.group('alias') or '（無）'}；"
        f"身分證：{m.group('twid') or '（無）'}"
    )
)

