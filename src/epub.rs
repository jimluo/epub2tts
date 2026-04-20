//! EPUB 解析器：基于成熟的 `epub` crate 读取 .epub，并抽取章节 XHTML/HTML 为纯文本段落。

use anyhow::{Context, Result};
use epub::doc::EpubDoc;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use time::OffsetDateTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterManifest {
    pub index: usize,
    pub title: String,
    pub text_hash: String,
    pub paragraphs: usize,
    pub output: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookManifest {
    pub tool: String,
    pub created_at: String,
    pub book_id: String,
    pub source: String,
    pub title: String,
    pub chapters: Vec<ChapterManifest>,
}

#[derive(Debug, Clone)]
pub struct Chapter {
    pub index: usize,
    pub title: String,
    pub paragraphs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EpubBook {
    pub title: String,
    pub chapters: Vec<Chapter>,
    pub source: PathBuf,
}

pub struct OpfInfo {
    pub rootfile: String,
    pub manifest_count: usize,
    pub spine_count: usize,
    pub title: String,
}

impl EpubBook {
    pub fn safe_book_id(&self) -> String {
        let mut id = self.title.trim().to_string();
        if id.is_empty() {
            id = self
                .source
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("book")
                .to_string();
        }
        let mut s = String::with_capacity(id.len());
        for c in id.chars() {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                s.push(c);
            } else {
                s.push('_');
            }
        }
        s.chars().filter(|c| !c.is_control()).collect()
    }

    pub fn to_manifest(&self) -> BookManifest {
        let chapters = self.chapters.iter().map(chapter_to_manifest).collect();
        BookManifest {
            tool: "epub2tts".to_string(),
            created_at: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_default(),
            book_id: self.safe_book_id(),
            source: self.source.to_string_lossy().to_string(),
            title: self.title.clone(),
            chapters,
        }
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let mut doc = EpubDoc::new(path).with_context(|| format!("open {}", path.display()))?;

        let title = get_book_title(&doc).unwrap_or_else(|| file_stem_or(path, "Book"));
        let chapters = read_chapters(&mut doc)?;

        Ok(Self {
            title,
            chapters,
            source: path.to_path_buf(),
        })
    }
}

pub fn dump_opf(path: &Path) -> Result<OpfInfo> {
    let doc = EpubDoc::new(path).with_context(|| format!("open {}", path.display()))?;
    let title = doc
        .mdata("title")
        .map(|m| m.value.to_string())
        .unwrap_or_default();
    Ok(OpfInfo {
        rootfile: "epub crate".to_string(),
        manifest_count: doc.resources.len(),
        spine_count: doc.spine.len(),
        title,
    })
}

pub fn debug_spine(path: &Path, head: usize) -> Result<()> {
    let doc = EpubDoc::new(path).with_context(|| format!("open {}", path.display()))?;
    println!("spine: {}", doc.spine.len());
    for (i, item) in doc.spine.iter().take(head).enumerate() {
        let idref = item.idref.to_string();
        let res = doc.resources.get(&idref);
        if let Some(r) = res {
            println!(
                "[{:02}] idref={} path={} mime={}",
                i,
                idref,
                r.path.display(),
                r.mime
            );
        } else {
            println!("[{:02}] idref={} missing in resources", i, idref);
        }
    }
    Ok(())
}

pub fn list_zip(path: &Path, head: usize) -> Result<()> {
    let doc = EpubDoc::new(path).with_context(|| format!("open {}", path.display()))?;
    println!("resources: {}", doc.resources.len());
    println!("spine: {}", doc.spine.len());

    let mut keys: Vec<String> = doc.resources.keys().cloned().collect();
    keys.sort();

    for (i, k) in keys.into_iter().take(head).enumerate() {
        if let Some(r) = doc.resources.get(&k) {
            println!(
                "[{:04}] {} | path={} | mime={}",
                i,
                k,
                r.path.display(),
                r.mime
            );
        }
    }
    Ok(())
}

fn extract_text(xhtml: &str) -> (String, Vec<String>) {
    let doc = Html::parse_document(xhtml);

    let sel_title = Selector::parse("title").ok();
    let mut title = String::new();
    if let Some(sel) = &sel_title
        && let Some(node) = doc.select(sel).next()
    {
        title = node.text().collect::<Vec<_>>().join("").trim().to_string();
    }

    let sel_h1 = Selector::parse("h1").ok();
    let sel_h2 = Selector::parse("h2").ok();
    let mut heading_title = String::new();
    if let Some(sel) = &sel_h1
        && let Some(node) = doc.select(sel).next()
    {
        let t = node.text().collect::<Vec<_>>().join("").trim().to_string();
        if !t.is_empty() {
            heading_title = t;
        }
    }
    if heading_title.is_empty()
        && let Some(sel) = &sel_h2
        && let Some(node) = doc.select(sel).next()
    {
        let t = node.text().collect::<Vec<_>>().join("").trim().to_string();
        if !t.is_empty() {
            heading_title = t;
        }
    }
    if title.is_empty() && !heading_title.is_empty() {
        title = heading_title.clone();
    }

    let sels = [
        sel_h1.clone(),
        sel_h2.clone(),
        Selector::parse("p").ok(),
        Selector::parse("li").ok(),
    ];

    let mut paragraphs = Vec::new();
    for s in sels.into_iter().flatten() {
        for node in doc.select(&s) {
            let t = node.text().collect::<Vec<_>>().join("");
            let norm = normalize_text(&t);
            if norm.is_empty() {
                continue;
            }
            let tag_name = node.value().name();
            let is_heading =
                tag_name.eq_ignore_ascii_case("h1") || tag_name.eq_ignore_ascii_case("h2");
            if !heading_title.is_empty() && is_heading && norm == heading_title {
                continue;
            }
            paragraphs.push(norm);
        }
    }

    if paragraphs.is_empty() {
        let body_sel = Selector::parse("body").ok();
        if let Some(sel) = body_sel
            && let Some(node) = doc.select(&sel).next()
        {
            let t = node.text().collect::<Vec<_>>().join("");
            let norm = normalize_text(&t);
            if !norm.is_empty() {
                paragraphs.push(norm);
            }
        }
    }

    (title, paragraphs)
}

fn chapter_to_manifest(c: &Chapter) -> ChapterManifest {
    ChapterManifest {
        index: c.index,
        title: c.title.clone(),
        text_hash: hash_text(&c.paragraphs.join("\n")),
        paragraphs: c.paragraphs.len(),
        output: None,
        status: "parsed".to_string(),
    }
}

fn get_book_title<R: Read + Seek>(doc: &EpubDoc<R>) -> Option<String> {
    doc.mdata("title")
        .map(|m| m.value.to_string())
        .filter(|s| !s.trim().is_empty())
}

fn file_stem_or(path: &Path, fallback: &str) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback)
        .to_string()
}

fn read_chapters<R: Read + Seek>(doc: &mut EpubDoc<R>) -> Result<Vec<Chapter>> {
    let spine_len = doc.spine.len();
    let mut chapters = Vec::with_capacity(spine_len);
    for idx in 0..spine_len {
        if let Some(ch) = read_chapter(doc, idx) {
            chapters.push(ch);
        }
    }
    Ok(chapters)
}

fn read_chapter<R: Read + Seek>(doc: &mut EpubDoc<R>, idx: usize) -> Option<Chapter> {
    if !doc.set_current_chapter(idx) {
        return None;
    }
    let (xhtml, _mime) = doc.get_current_str().unwrap_or_default();
    let (chapter_title, paragraphs) = extract_text(&xhtml);
    let final_title = choose_chapter_title(chapter_title, doc.get_current_id(), &paragraphs);
    Some(Chapter {
        index: idx,
        title: final_title,
        paragraphs,
    })
}

fn is_placeholder_title(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    matches!(
        t.to_ascii_lowercase().as_str(),
        "chapter" | "unknown" | "untitled"
    ) || t == "未知"
}

fn paragraph_title_fallback(paragraphs: &[String]) -> Option<String> {
    let p = paragraphs.first()?.trim();
    if p.is_empty() {
        return None;
    }
    let mut cut = String::new();
    for ch in p.chars().take(32) {
        cut.push(ch);
    }
    let trimmed = cut
        .trim()
        .trim_matches(|c: char| {
            c.is_whitespace() || matches!(c, '。' | '，' | ',' | '.' | '；' | ';' | '：' | ':')
        })
        .to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn choose_chapter_title(
    extracted_title: String,
    current_id: Option<String>,
    paragraphs: &[String],
) -> String {
    if !is_placeholder_title(&extracted_title) {
        return extracted_title;
    }
    if let Some(id) = current_id {
        if !is_placeholder_title(&id) {
            return id;
        }
    }
    if let Some(t) = paragraph_title_fallback(paragraphs) {
        return t;
    }
    "Chapter".to_string()
}

fn normalize_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        let is_space = ch.is_whitespace();
        if is_space {
            if !prev_space {
                out.push(' ');
            }
        } else {
            out.push(ch);
        }
        prev_space = is_space;
    }
    out.trim().to_string()
}

fn hash_text(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_should_fallback_to_heading_when_title_missing() {
        let html = r#"
            <html>
              <head></head>
              <body>
                <h1>第一章</h1>
                <p>内容。</p>
              </body>
            </html>
        "#;
        let (title, paragraphs) = extract_text(html);
        assert_eq!(title, "第一章");
        assert_eq!(paragraphs, vec!["内容。"]);
    }

    #[test]
    fn safe_book_id_should_be_stable() {
        let b = EpubBook {
            title: "test".to_string(),
            chapters: vec![],
            source: PathBuf::from("x.epub"),
        };
        assert_eq!(b.safe_book_id(), "test");
    }

    #[test]
    fn file_stem_or_should_fallback_when_missing() {
        assert_eq!(file_stem_or(Path::new("a.epub"), "Book"), "a");
        assert_eq!(file_stem_or(Path::new("a"), "Book"), "a");
    }

    #[test]
    fn chapter_to_manifest_should_hash_and_count_paragraphs() {
        let c = Chapter {
            index: 0,
            title: "t".to_string(),
            paragraphs: vec!["a".to_string(), "b".to_string()],
        };
        let m = chapter_to_manifest(&c);
        assert_eq!(m.index, 0);
        assert_eq!(m.title, "t");
        assert_eq!(m.paragraphs, 2);
        assert_eq!(m.status, "parsed");
        assert!(!m.text_hash.is_empty());
    }
}
