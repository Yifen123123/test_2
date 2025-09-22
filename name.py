def get_name_pattern(name_list): 
    # 1. 把姓氏 join 起來，變成 (王|李|張)
    name_pattern = r"(" + r"|".join(map(re.escape, name_list)) + r")[\u4E00-\u9FFF]{1,2}"

    # 2. 別名（即 / 原名 / 又名 / 曾用名），整段可選
    alias_alt = r"(?:%s)[\u4E00-\u9FFF]{1,2}" % "|".join(map(re.escape, name_list))
    alias_block = rf"(?:即\s*(?:{alias_alt})|（(?:原名|又名|曾用名)[:：]?\s*(?:{alias_alt})）)?"

    # 3. 身分證，可選（0–20 個字之內）
    twid_block = r"[\s\S]{0,20}([A-Z][0-9]{9})?"

    # 4. 拼接完整 regex
    name_pattern = name_pattern + alias_block + twid_block
    return name_pattern
