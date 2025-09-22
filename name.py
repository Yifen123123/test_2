# 支援二字或三字姓名（單姓 + 1~2 個字）
alias_alt = r"(?:%s)[\u4E00-\u9FFF]{1,2}" % "|".join(map(re.escape, name_list))

# 別名（即 / 原名 / 又名 / 曾用名），整段可選
alias_block = rf"(?:即\s*(?:{alias_alt})|（(?:原名|又名|曾用名)[:：]?\s*(?:{alias_alt})）)?"

# 身分證，可選
twid_block = r"[\s\S]{0,20}([A-Z][0-9]{9})?"

# 拼接到原本的 name_pattern
name_pattern = name_pattern + alias_block + twid_block
