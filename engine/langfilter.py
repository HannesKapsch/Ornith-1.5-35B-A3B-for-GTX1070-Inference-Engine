"""Language filter settings for the output head (PLAN_SPRACHFILTER.md).

Each switchable group (cjk, other) is 'auto', 'on' or 'off'. 'auto' turns the group on for the next answer as soon as
the conversation (user text, pasted text, tool results / web pages) contains a character of that script.
latin (with Greek, for math symbols) is always on. Only the output side is filtered; the model still reads every script."""
import re

GROUPS = ('cjk', 'other')
MODES = ('auto', 'on', 'off')

_CJK = re.compile('[぀-ヿ㄀-ㄯ㄰-㆏ㆠ-ㇿ㐀-䶿一-鿿ꥠ-꥿'
                  '가-퟿豈-﫿ｦ-ﾟ\U00020000-\U0003ffff]')
_OTHER = re.compile('[Ѐ-ԯᲀ-᲏ⷠ-ⷿꙀ-ꚟ԰-֏'
                    '֐-׿؀-ۿ܀-ࣿऀ-෿฀-໿ༀ-࿿က-႟'
                    'Ⴀ-ჿሀ-᎟ក-៿יִ-﷿ﹰ-﻿]')


def present(text):
    """which switchable scripts occur in text"""
    return dict(cjk=bool(_CJK.search(text)), other=bool(_OTHER.search(text)))


def resolve(settings, text):
    """settings {'cjk': mode, 'other': mode} + conversation text -> {'cjk': bool, 'other': bool} for set_langs"""
    seen = present(text) if any(settings.get(g, 'auto') == 'auto' for g in GROUPS) else {}
    out = {}
    for g in GROUPS:
        m = settings.get(g, 'auto')
        out[g] = True if m == 'on' else False if m == 'off' else seen[g]
    return out
