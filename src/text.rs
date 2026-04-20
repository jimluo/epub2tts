//! 中文小说向 TTS 文本处理：规范化（空白/标点/数字）与分句分段。

use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy)]
pub struct SegmentConfig {
    pub max_chars_per_chunk: usize,
    pub numbers_to_zh: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_chars_per_chunk: 160,
            numbers_to_zh: true,
        }
    }
}

fn normalize_for_tts_with_numbers(input: &str, numbers_to_zh: bool) -> String {
    let s = input.nfkc().collect::<String>();
    let s = normalize_whitespace(&s);
    let s = normalize_punctuation_zh(&s);
    if numbers_to_zh {
        convert_numbers_zh(&s)
    } else {
        s
    }
}

#[cfg(test)]
fn normalize_for_tts(input: &str) -> String {
    normalize_for_tts_with_numbers(input, true)
}

fn normalize_punctuation_zh(s: &str) -> String {
    s.chars()
        .map(|ch| match ch {
            '!' => '！',
            '?' => '？',
            ';' => '；',
            ':' => '：',
            _ => ch,
        })
        .collect()
}

pub fn segment_paragraphs(paragraphs: &[String], cfg: SegmentConfig) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for p in paragraphs {
        let p = normalize_for_tts_with_numbers(p, cfg.numbers_to_zh);
        if p.is_empty() {
            continue;
        }
        for sent in split_sentences_zh(&p) {
            if sent.is_empty() {
                continue;
            }
            if char_count(&sent) > cfg.max_chars_per_chunk {
                for part in split_long_sentence(&sent, cfg.max_chars_per_chunk) {
                    push_chunk(&mut out, &mut current, &part, cfg.max_chars_per_chunk);
                }
                continue;
            }
            push_chunk(&mut out, &mut current, &sent, cfg.max_chars_per_chunk);
        }
    }

    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }

    out
}

fn push_chunk(out: &mut Vec<String>, current: &mut String, piece: &str, max_chars: usize) {
    if current.is_empty() {
        current.push_str(piece.trim());
        return;
    }
    let candidate_len = char_count(current) + 1 + char_count(piece);
    if candidate_len <= max_chars {
        current.push(' ');
        current.push_str(piece.trim());
        return;
    }
    out.push(current.trim().to_string());
    current.clear();
    current.push_str(piece.trim());
}

fn split_sentences_zh(s: &str) -> Vec<String> {
    let sentences = sentencex::segment("zh", s);
    let out: Vec<String> = sentences
        .into_iter()
        .map(|t| t.trim().to_string())
        .collect();
    if out.iter().all(|s| s.is_empty()) {
        return vec![s.trim().to_string()];
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

fn split_long_sentence(s: &str, max_chars: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        cur.push(ch);
        if char_count(&cur) >= max_chars {
            let cut = find_last_soft_break(&cur).unwrap_or(char_count(&cur));
            let (left, right) = split_at_char(&cur, cut);
            let left = left.trim().to_string();
            if !left.is_empty() {
                parts.push(left);
            }
            cur = right.trim().to_string();
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }
    parts
}

fn find_last_soft_break(s: &str) -> Option<usize> {
    let mut last = None;
    let mut idx = 0usize;
    for ch in s.chars() {
        idx += 1;
        if matches!(ch, '，' | ',' | '、' | '；' | ';' | '：' | ':') {
            last = Some(idx);
        }
    }
    last
}

fn split_at_char(s: &str, char_idx: usize) -> (&str, &str) {
    if char_idx == 0 {
        return ("", s);
    }
    let byte_idx = s
        .char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    (&s[..byte_idx], &s[byte_idx..])
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn convert_numbers_zh(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let digits: String = chars[start..i].iter().collect();
            if i < chars.len() && chars[i] == '.' {
                let dot = i;
                let mut j = dot + 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                if j > dot + 1 {
                    let frac: String = chars[dot + 1..j].iter().collect();
                    out.push_str(&number_to_zh(&digits));
                    out.push('点');
                    for d in frac.chars() {
                        out.push_str(digit_to_zh(d));
                    }
                    i = j;
                    continue;
                }
            }
            if i < chars.len() && chars[i] == '%' {
                out.push_str("百分之");
                out.push_str(&number_to_zh(&digits));
                i += 1;
                continue;
            }
            if digits.len() == 4 && i < chars.len() && chars[i] == '年' {
                for d in digits.chars() {
                    out.push_str(digit_to_zh(d));
                }
                out.push('年');
                i += 1;
                continue;
            }
            out.push_str(&number_to_zh(&digits));
            continue;
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn number_to_zh(digits: &str) -> String {
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return "零".to_string();
    }
    if digits.len() > 16 {
        return digits.chars().map(digit_to_zh).collect();
    }
    let groups = split_into_groups_of_four(digits);
    let mut out = String::new();
    for (idx, raw) in groups.iter().enumerate() {
        let big_unit = big_unit_for(groups.len() - idx - 1);
        let p = group_0_9999_to_zh(raw);
        if p.is_empty() {
            continue;
        }
        if idx > 0
            && raw.len() == 4
            && raw.starts_with('0')
            && !out.is_empty()
            && !out.ends_with('零')
        {
            out.push('零');
        }
        out.push_str(&p);
        out.push_str(big_unit);
    }
    while out.ends_with('零') {
        out.pop();
    }
    out
}

fn split_into_groups_of_four(digits: &str) -> Vec<String> {
    let mut groups = Vec::new();
    let mut rem = digits.to_string();
    while !rem.is_empty() {
        let take = rem.len().min(4);
        let split = rem.len() - take;
        let g = rem[split..].to_string();
        rem.truncate(split);
        groups.push(g);
    }
    groups.reverse();
    groups
}

fn big_unit_for(idx: usize) -> &'static str {
    match idx {
        0 => "",
        1 => "万",
        2 => "亿",
        3 => "兆",
        _ => "",
    }
}

fn group_0_9999_to_zh(g: &str) -> String {
    let mut digits: Vec<u32> = g.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.is_empty() {
        return String::new();
    }
    while digits.len() < 4 {
        digits.insert(0, 0);
    }
    let units = ["千", "百", "十", ""];
    let mut out = String::new();
    let mut zero_pending = false;
    for (i, d) in digits.iter().enumerate() {
        let is_last = i == digits.len() - 1;
        if *d == 0 {
            if !out.is_empty() && !is_last {
                zero_pending = true;
            }
            continue;
        }
        if zero_pending {
            out.push('零');
            zero_pending = false;
        }
        if *d == 1 && units[i] == "十" && out.is_empty() {
            out.push('十');
        } else {
            out.push_str(digit_to_zh_char(*d));
            out.push_str(units[i]);
        }
    }
    out
}

fn digit_to_zh(d: char) -> &'static str {
    match d {
        '0' => "零",
        '1' => "一",
        '2' => "二",
        '3' => "三",
        '4' => "四",
        '5' => "五",
        '6' => "六",
        '7' => "七",
        '8' => "八",
        '9' => "九",
        _ => "",
    }
}

fn digit_to_zh_char(d: u32) -> &'static str {
    match d {
        0 => "零",
        1 => "一",
        2 => "二",
        3 => "三",
        4 => "四",
        5 => "五",
        6 => "六",
        7 => "七",
        8 => "八",
        9 => "九",
        _ => "",
    }
}

fn char_count(s: &str) -> usize {
    s.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_should_collapse_whitespace() {
        assert_eq!(normalize_for_tts("  你好 \n 世界\t"), "你好 世界");
    }

    #[test]
    fn numbers_should_convert_percent_and_decimal() {
        assert_eq!(normalize_for_tts("50%"), "百分之五十");
        assert_eq!(normalize_for_tts("3.14"), "三点一四");
    }

    #[test]
    fn numbers_should_convert_year_as_digits() {
        assert_eq!(normalize_for_tts("2024年"), "二零二四年");
    }

    #[test]
    fn numbers_should_convert_integers() {
        assert_eq!(normalize_for_tts("0"), "零");
        assert_eq!(normalize_for_tts("10"), "十");
        assert_eq!(normalize_for_tts("11"), "十一");
        assert_eq!(normalize_for_tts("20"), "二十");
        assert_eq!(normalize_for_tts("105"), "一百零五");
        assert_eq!(normalize_for_tts("1005"), "一千零五");
        assert_eq!(normalize_for_tts("10000"), "一万");
        assert_eq!(normalize_for_tts("10001"), "一万零一");
        assert_eq!(normalize_for_tts("12345"), "一万二千三百四十五");
    }

    #[test]
    fn segmentation_should_split_by_sentence_end() {
        let paragraphs = vec![String::from("你好。世界！再见？")];
        let chunks = segment_paragraphs(
            &paragraphs,
            SegmentConfig {
                max_chars_per_chunk: 100,
                numbers_to_zh: true,
            },
        );
        assert_eq!(chunks, vec!["你好。 世界！ 再见？"]);
    }

    #[test]
    fn segmentation_should_split_long_sentence() {
        let paragraphs = vec![String::from(
            "一二三四五六七八九十，一二三四五六七八九十，一二三四五六七八九十。",
        )];
        let chunks = segment_paragraphs(
            &paragraphs,
            SegmentConfig {
                max_chars_per_chunk: 12,
                numbers_to_zh: true,
            },
        );
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| char_count(c) <= 12));
    }
}
