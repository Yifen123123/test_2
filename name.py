alias_alt = r"(?:%s)[\u4E00-\u9FFF]{1,2}" % "|".join(map(re.escape, name_list))

# 尾巴改成：可選別名區塊 + 間距 + 身分證
name_pattern = name_pattern + rf"(?:\s*(?:即\s*(?:{alias_alt})|（(?:原名|又名|曾用名)[:：]?\s*(?:{alias_alt})）))?[\s\S]{{0,20}}([A-Z][0-9]{{9}})"