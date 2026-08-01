"""国家/组织代码的统一中文显示标签。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# 兼容 SATCAT 专用代码、常见 ISO 代码及国际航天组织代码。
COUNTRY_OR_ORG_LABELS = {
    "AE": "阿联酋",
    "ALG": "阿尔及利亚",
    "AQ": "南极",
    "AR": "阿根廷",
    "ARGN": "阿根廷",
    "AS": "澳大利亚",
    "AU": "澳大利亚",
    "AUS": "澳大利亚",
    "AUST": "奥地利",
    "BE": "比利时",
    "BEL": "比利时",
    "BIOT": "英属印度洋领地",
    "BR": "巴西",
    "BRAZ": "巴西",
    "CA": "加拿大",
    "CHLE": "智利",
    "CL": "智利",
    "CIS": "俄罗斯",
    "CN": "中国",
    "CZ": "捷克",
    "CZCH": "捷克",
    "DE": "德国",
    "DEN": "丹麦",
    "DK": "丹麦",
    "EGYP": "埃及",
    "ES": "西班牙",
    "ESA": "欧洲航天局",
    "ESRO": "欧洲空间研究组织",
    "ET": "埃塞俄比亚",
    "EU": "欧盟/欧洲组织",
    "EUTE": "欧洲气象卫星应用组织（EUMETSAT）",
    "FI": "芬兰",
    "FIN": "芬兰",
    "FR": "法国",
    "GB": "英国",
    "GER": "德国",
    "GL": "格陵兰",
    "GLOB": "国际/全球组织",
    "GR": "希腊",
    "GREC": "希腊",
    "HU": "匈牙利",
    "ID": "印度尼西亚",
    "IN": "印度",
    "IND": "印度",
    "INDO": "印度尼西亚",
    "IR": "伊朗",
    "IRAN": "伊朗",
    "ISRA": "以色列",
    "IT": "意大利",
    "ITA": "意大利",
    "JP": "日本",
    "JPN": "日本",
    "KAZ": "哈萨克斯坦",
    "KR": "韩国",
    "LU": "卢森堡",
    "LUXE": "卢森堡",
    "MALA": "马来西亚",
    "MEX": "墨西哥",
    "MU": "毛里求斯",
    "MY": "马来西亚",
    "NETH": "荷兰",
    "NG": "尼日利亚",
    "NIG": "尼日利亚",
    "NKOR": "朝鲜",
    "NL": "荷兰",
    "NO": "挪威",
    "NOR": "挪威",
    "NZ": "新西兰",
    "PH": "菲律宾",
    "PHIL": "菲律宾",
    "PL": "波兰",
    "POL": "波兰",
    "POR": "葡萄牙",
    "PRC": "中国",
    "PT": "葡萄牙",
    "RO": "罗马尼亚",
    "RP": "菲律宾",
    "RU": "俄罗斯",
    "SAFR": "南非",
    "SAUD": "沙特阿拉伯",
    "SE": "瑞典",
    "SES": "SES（卢森堡卫星运营商）",
    "SG": "新加坡",
    "SING": "新加坡",
    "SKOR": "韩国",
    "SPN": "西班牙",
    "SW": "瑞典",
    "SWED": "瑞典",
    "SWTZ": "瑞士",
    "THAI": "泰国",
    "TR": "土耳其",
    "TURK": "土耳其",
    "TW": "中国台湾",
    "TWN": "中国台湾",
    "UA": "乌克兰",
    "UAE": "阿联酋",
    "UK": "英国",
    "UKR": "乌克兰",
    "US": "美国",
    "USA": "美国",
    "VTNM": "越南",
    "WS": "萨摩亚",
    "ZA": "南非",
}


def country_or_org_label(code: Any) -> str:
    """将国家/组织代码转换为中文标签，未知代码保留原值以便核查。"""
    if code is None:
        return "未知"
    raw = str(code).strip()
    if not raw or raw.lower() in {"?", "nan", "<na>", "nat", "none"}:
        return "未知"
    return COUNTRY_OR_ORG_LABELS.get(raw.upper(), raw)


def add_country_or_org_column(
    df: pd.DataFrame,
    code_column: str = "国家代码",
    label_column: str = "国家/组织",
) -> pd.DataFrame:
    """复制 DataFrame，并在代码列后插入对应的国家/组织标签列。"""
    if code_column not in df.columns:
        return df

    result = df.copy()
    labels = result[code_column].map(country_or_org_label)
    if label_column in result.columns:
        result[label_column] = labels
        return result

    code_position = result.columns.get_loc(code_column)
    result.insert(code_position + 1, label_column, labels)
    return result
