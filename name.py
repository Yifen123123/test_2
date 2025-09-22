import re

def get_name_list():
    name_list = []
    with open("name_deleted.txt", "r", encoding="utf-8") as file:
        for line in file:
            name_list.extend(line.strip().split(","))
    return name_list

def get_name_pattern(name_list):
    # 1) 名單做 regex-escape，並以長度由大到小避免子字串搶匹配
    escaped = sorted((re.escape(n) for n in name_list if n), key=len, reverse=True)
    alt     = r"(?:%s)" % "|".join(escaped)

    # 2) 主名 + 可選的別名區塊（即/原名/又名/曾用名）
    alias_block = rf"(?:\s*(?:即|（(?:原名|又名|曾用名)[:：]?\s*)(?P<alias>{alt})[）)]?)?"

    # 3) 身分證改成可選（?），中間允許 0~10 任意字元
    twid = r"(?:[\s\S]{0,10}(?P<twid>[A-Z][0-9]{9}))?"

    # 4) 最終：主名 (name) + 別名區塊 + 可選身分證
    name_pattern = rf"(?P<name>{alt}){alias_block}{twid}"
    return name_pattern
