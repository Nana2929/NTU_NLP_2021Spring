import ipdb


def normalize_text(example):
    """把字串全形轉半形"""
    """Lowercase all the text"""
    example['text'] = example['text'].replace("。", ".").replace("、", ',')
    rstring = ""
    for uchar in example['text']:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    example['text'] = rstring.lower()
    return example


if __name__ == '__main__':
    s = "ＡＢＣＤ。？"
    a = normalize_text(s)
    print(a)
