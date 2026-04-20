//! epub2tts CLI 入口：负责命令行参数解析与任务调度。

mod audio;
mod epub;
mod text;
mod tts;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

#[derive(Parser)]
#[command(name = "epub2tts")]
#[command(about = "EPUB to TTS pipeline (v1: CLI + EPUB parsing)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    Convert {
        input: PathBuf,
        #[arg(long, default_value = "out")]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        resume: bool,
        #[arg(long, default_value = "fish")]
        engine: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long, default_value = "1")]
        jobs: String,
    },
    Inspect {
        input: PathBuf,
        #[arg(long, default_value_t = 5)]
        head: usize,
    },
    DumpOpf {
        input: PathBuf,
    },
    DebugSpine {
        input: PathBuf,
        #[arg(long, default_value_t = 16)]
        head: usize,
    },
    ListZip {
        input: PathBuf,
        #[arg(long, default_value_t = 50)]
        head: usize,
    },
}

struct ConvertOptions<'a> {
    out: &'a std::path::Path,
    engine: &'a str,
    jobs: &'a str,
    resume: bool,
    model_dir: Option<&'a std::path::Path>,
    voice: Option<&'a str>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Convert {
            input,
            out,
            engine,
            jobs,
            resume,
            model_dir,
            voice,
            ..
        } => run_convert(
            &input,
            ConvertOptions {
                out: &out,
                engine: &engine,
                jobs: &jobs,
                resume,
                model_dir: model_dir.as_deref(),
                voice: voice.as_deref(),
            },
        )?,
        Commands::Inspect { input, head } => {
            let book = epub::EpubBook::from_path(&input)?;
            println!("title: {}", book.title);
            println!("chapters: {}", book.chapters.len());
            for ch in book.chapters.iter().take(head) {
                let chunks =
                    text::segment_paragraphs(&ch.paragraphs, text::SegmentConfig::default());
                println!(
                    "[{:03}] {} | paragraphs: {} | chunks: {}",
                    ch.index + 1,
                    ch.title,
                    ch.paragraphs.len(),
                    chunks.len()
                );
            }
        }
        Commands::DumpOpf { input } => {
            let info = epub::dump_opf(&input)?;
            println!("rootfile: {}", info.rootfile);
            println!("manifest_items: {}", info.manifest_count);
            println!("spine_items: {}", info.spine_count);
            println!("title: {}", info.title);
        }
        Commands::DebugSpine { input, head } => {
            epub::debug_spine(&input, head)?;
        }
        Commands::ListZip { input, head } => {
            epub::list_zip(&input, head)?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct EngineOptions {
    engine: String,
    model_dir: Option<PathBuf>,
    voice: Option<String>,
}

#[derive(Clone)]
struct ChapterJob {
    index: usize,
    chapter: epub::Chapter,
}

struct ChapterResult {
    index: usize,
    title: String,
    out_path: Option<PathBuf>,
    error: Option<String>,
}

fn run_convert(input: &std::path::Path, opts: ConvertOptions<'_>) -> anyhow::Result<()> {
    let book = epub::EpubBook::from_path(input)?;
    let book_id = book.safe_book_id();
    let out_dir = opts.out.join(&book_id);
    std::fs::create_dir_all(&out_dir)?;

    let chapters_dir = out_dir.join("chapters");
    std::fs::create_dir_all(&chapters_dir)?;

    let manifest_path = out_dir.join("manifest.json");
    let mut manifest = load_or_new_manifest(&manifest_path, &book, opts.resume)?;
    let engine_opts = EngineOptions {
        engine: opts.engine.to_string(),
        model_dir: opts.model_dir.map(|p| p.to_path_buf()),
        voice: opts.voice.map(|s| s.to_string()),
    };
    let tts = build_tts_engine_from_options(&engine_opts)?;
    let numbers_to_zh = opts.engine != "kokoro" && opts.engine != "fish";
    let max_chars_per_chunk = match opts.engine {
        "kokoro" => 80,
        _ => 160,
    };
    let sample_rate = tts.sample_rate();
    drop(tts);
    let pause = audio::silence_i16(sample_rate, 150);
    let total = book.chapters.len();
    let jobs = resolve_jobs(opts.jobs, total)?;
    if jobs == 1 {
        let mut tts = build_tts_engine_from_options(&engine_opts)?;
        for (i, ch) in book.chapters.iter().enumerate() {
            process_single_chapter(
                &mut manifest,
                &mut *tts,
                sample_rate,
                &pause,
                numbers_to_zh,
                max_chars_per_chunk,
                &chapters_dir,
                total,
                opts.resume,
                i,
                ch,
            );
        }
    } else {
        process_chapters_parallel(
            &mut manifest,
            &book,
            &engine_opts,
            sample_rate,
            &pause,
            numbers_to_zh,
            max_chars_per_chunk,
            &chapters_dir,
            total,
            opts.resume,
            jobs,
        )?;
    }

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!("parsed: {} chapters", book.chapters.len());
    println!("audio: {}", chapters_dir.display());
    println!("manifest: {}", manifest_path.display());
    Ok(())
}

fn resolve_jobs(spec: &str, total: usize) -> anyhow::Result<usize> {
    let cap = total.max(1);
    if spec.eq_ignore_ascii_case("auto") {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        return Ok(cores.clamp(1, cap));
    }
    let parsed = spec
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("--jobs must be a positive integer or 'auto', got: {spec}"))?;
    if parsed == 0 {
        anyhow::bail!("--jobs must be >= 1 (or use --jobs auto)");
    }
    Ok(parsed.min(cap))
}

#[allow(clippy::too_many_arguments)]
fn process_single_chapter(
    manifest: &mut epub::BookManifest,
    tts: &mut dyn tts::TtsEngine,
    sample_rate: u32,
    pause: &[i16],
    numbers_to_zh: bool,
    max_chars_per_chunk: usize,
    chapters_dir: &std::path::Path,
    total: usize,
    resume: bool,
    i: usize,
    ch: &epub::Chapter,
) {
    let idx1 = i + 1;
    let percent = (idx1 as f64 / total as f64 * 100.0).round() as u32;
    if resume && should_skip_chapter(manifest, i) {
        println!("[{}/{} {:>3}%] 跳过 {}", idx1, total, percent, ch.title);
        return;
    }

    println!("[{}/{} {:>3}%] 合成 {}", idx1, total, percent, ch.title);
    let out_path = match synthesize_chapter_wav(
        tts,
        sample_rate,
        pause,
        numbers_to_zh,
        max_chars_per_chunk,
        chapters_dir,
        i,
        ch,
    ) {
        Ok(p) => p,
        Err(e) => {
            println!(
                "[{}/{} {:>3}%] 失败 {}: {}",
                idx1, total, percent, ch.title, e
            );
            if let Some(m) = manifest.chapters.get_mut(i) {
                m.output = None;
                m.status = "error".to_string();
            }
            return;
        }
    };

    let duration_s = wav_duration_seconds(&out_path, sample_rate);
    println!(
        "[{}/{} {:>3}%] 完成 {} {:.1}s -> {}",
        idx1,
        total,
        percent,
        ch.title,
        duration_s,
        out_path.display()
    );
    if let Some(m) = manifest.chapters.get_mut(i) {
        m.output = Some(out_path.to_string_lossy().to_string());
        m.status = "done".to_string();
    }
}

#[allow(clippy::too_many_arguments)]
fn process_chapters_parallel(
    manifest: &mut epub::BookManifest,
    book: &epub::EpubBook,
    engine_opts: &EngineOptions,
    sample_rate: u32,
    pause: &[i16],
    numbers_to_zh: bool,
    max_chars_per_chunk: usize,
    chapters_dir: &std::path::Path,
    total: usize,
    resume: bool,
    jobs: usize,
) -> anyhow::Result<()> {
    let mut pending = Vec::new();
    for (i, ch) in book.chapters.iter().enumerate() {
        let idx1 = i + 1;
        let percent = (idx1 as f64 / total as f64 * 100.0).round() as u32;
        if resume && should_skip_chapter(manifest, i) {
            println!("[{}/{} {:>3}%] 跳过 {}", idx1, total, percent, ch.title);
            continue;
        }
        pending.push(ChapterJob {
            index: i,
            chapter: ch.clone(),
        });
    }
    if pending.is_empty() {
        return Ok(());
    }

    println!(
        "并行合成：{} workers / {} chapters",
        jobs.min(pending.len()),
        pending.len()
    );
    let work = Arc::new(pending);
    let cursor = Arc::new(AtomicUsize::new(0));
    let pause = Arc::new(pause.to_vec());
    let chapters_dir = chapters_dir.to_path_buf();
    let (tx, rx) = mpsc::channel::<ChapterResult>();

    #[cfg(feature = "fish")]
    let fish_shared = if engine_opts.engine == "fish" {
        let model_dir = engine_opts
            .model_dir
            .as_deref()
            .unwrap_or_else(|| std::path::Path::new("./fish"));
        let voice = engine_opts.voice.as_deref().unwrap_or("default");
        Some(Arc::new(tts::FishSpeechShared::from_dir(model_dir, voice)?))
    } else {
        None
    };

    std::thread::scope(|scope| {
        for _ in 0..jobs.min(work.len()) {
            let tx = tx.clone();
            let work = Arc::clone(&work);
            let cursor = Arc::clone(&cursor);
            let pause = Arc::clone(&pause);
            let engine_opts = engine_opts.clone();
            let chapters_dir = chapters_dir.clone();
            #[cfg(feature = "fish")]
            let fish_shared = fish_shared.clone();
            scope.spawn(move || {
                let mut tts = match (|| -> anyhow::Result<Box<dyn tts::TtsEngine>> {
                    #[cfg(feature = "fish")]
                    if let Some(shared) = fish_shared {
                        let voice = engine_opts.voice.as_deref().unwrap_or("default");
                        return Ok(Box::new(tts::FishSpeechEngine::from_shared(
                            (*shared).clone(),
                            voice,
                        )?));
                    }
                    build_tts_engine_from_options(&engine_opts)
                })() {
                    Ok(tts) => tts,
                    Err(e) => {
                        let _ = tx.send(ChapterResult {
                            index: usize::MAX,
                            title: "engine".to_string(),
                            out_path: None,
                            error: Some(e.to_string()),
                        });
                        return;
                    }
                };
                loop {
                    let pos = cursor.fetch_add(1, Ordering::SeqCst);
                    if pos >= work.len() {
                        break;
                    }
                    let job = &work[pos];
                    let result = synthesize_chapter_wav(
                        &mut *tts,
                        sample_rate,
                        &pause,
                        numbers_to_zh,
                        max_chars_per_chunk,
                        &chapters_dir,
                        job.index,
                        &job.chapter,
                    );
                    let msg = match result {
                        Ok(path) => ChapterResult {
                            index: job.index,
                            title: job.chapter.title.clone(),
                            out_path: Some(path),
                            error: None,
                        },
                        Err(e) => ChapterResult {
                            index: job.index,
                            title: job.chapter.title.clone(),
                            out_path: None,
                            error: Some(e.to_string()),
                        },
                    };
                    let _ = tx.send(msg);
                }
            });
        }
        drop(tx);
        for result in rx {
            if result.index == usize::MAX {
                anyhow::bail!(
                    result
                        .error
                        .unwrap_or_else(|| "worker init failed".to_string())
                );
            }
            let idx1 = result.index + 1;
            let percent = (idx1 as f64 / total as f64 * 100.0).round() as u32;
            match (result.out_path, result.error) {
                (Some(out_path), None) => {
                    let duration_s = wav_duration_seconds(&out_path, sample_rate);
                    println!(
                        "[{}/{} {:>3}%] 完成 {} {:.1}s -> {}",
                        idx1,
                        total,
                        percent,
                        result.title,
                        duration_s,
                        out_path.display()
                    );
                    if let Some(m) = manifest.chapters.get_mut(result.index) {
                        m.output = Some(out_path.to_string_lossy().to_string());
                        m.status = "done".to_string();
                    }
                }
                (_, Some(error)) => {
                    println!(
                        "[{}/{} {:>3}%] 失败 {}: {}",
                        idx1, total, percent, result.title, error
                    );
                    if let Some(m) = manifest.chapters.get_mut(result.index) {
                        m.output = None;
                        m.status = "error".to_string();
                    }
                }
                _ => {}
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

fn load_or_new_manifest(
    manifest_path: &std::path::Path,
    book: &epub::EpubBook,
    resume: bool,
) -> anyhow::Result<epub::BookManifest> {
    if resume && manifest_path.exists() {
        let raw = std::fs::read_to_string(manifest_path)?;
        let existing: epub::BookManifest = serde_json::from_str(&raw)?;
        if existing.chapters.len() == book.chapters.len() {
            return Ok(existing);
        }
    }
    Ok(book.to_manifest())
}

fn should_skip_chapter(manifest: &epub::BookManifest, chapter_index: usize) -> bool {
    let Some(ch) = manifest.chapters.get(chapter_index) else {
        return false;
    };
    if ch.status != "done" {
        return false;
    }
    let Some(path) = &ch.output else {
        return false;
    };
    file_exists_and_nonempty(path)
}

fn file_exists_and_nonempty(path: &str) -> bool {
    let min_len = if path.to_ascii_lowercase().ends_with(".wav") {
        44
    } else {
        0
    };
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > min_len)
        .unwrap_or(false)
}

fn wav_duration_seconds(path: &std::path::Path, sample_rate: u32) -> f64 {
    std::fs::metadata(path)
        .ok()
        .map(|m| m.len())
        .and_then(|len| len.checked_sub(44))
        .map(|data| data as f64 / 2.0 / sample_rate as f64)
        .unwrap_or(0.0)
}

fn build_tts_engine(
    engine: &str,
    model_dir: Option<&std::path::Path>,
    voice: Option<&str>,
) -> anyhow::Result<Box<dyn tts::TtsEngine>> {
    #[cfg(not(feature = "fish"))]
    {
        let _ = model_dir;
        let _ = voice;
    }
    #[cfg(not(feature = "kokoro"))]
    {
        let _ = model_dir;
        let _ = voice;
    }
    match engine {
        #[cfg(feature = "fish")]
        "fish" => {
            let dir = model_dir.unwrap_or_else(|| std::path::Path::new("./fish"));
            let voice_name = voice.unwrap_or("default");
            Ok(Box::new(tts::FishSpeechEngine::from_dir(dir, voice_name)?))
        }
        #[cfg(not(feature = "fish"))]
        "fish" => anyhow::bail!("engine fish requires building with --features fish"),
        #[cfg(feature = "kokoro")]
        "kokoro" => {
            let dir = model_dir.unwrap_or_else(|| std::path::Path::new("./kokoro"));
            let voice_name = voice.unwrap_or("auto");
            Ok(Box::new(tts::KokoroEngine::from_dir(dir, voice_name)?))
        }
        #[cfg(not(feature = "kokoro"))]
        "kokoro" => anyhow::bail!("engine kokoro requires building with --features kokoro"),
        other => anyhow::bail!("engine not supported yet: {other} (use --engine fish|kokoro)"),
    }
}

fn build_tts_engine_from_options(opts: &EngineOptions) -> anyhow::Result<Box<dyn tts::TtsEngine>> {
    build_tts_engine(
        &opts.engine,
        opts.model_dir.as_deref(),
        opts.voice.as_deref(),
    )
}

fn synthesize_chapter_wav(
    tts: &mut dyn tts::TtsEngine,
    sample_rate: u32,
    pause: &[i16],
    numbers_to_zh: bool,
    max_chars_per_chunk: usize,
    chapters_dir: &std::path::Path,
    chapter_index: usize,
    ch: &epub::Chapter,
) -> anyhow::Result<std::path::PathBuf> {
    let cfg = text::SegmentConfig {
        max_chars_per_chunk,
        numbers_to_zh,
        ..Default::default()
    };
    let chunks = text::segment_paragraphs(&ch.paragraphs, cfg);
    let mut pcm = Vec::new();
    for (ci, chunk) in chunks.iter().enumerate() {
        let mut part = tts.synthesize(chunk)?;
        pcm.append(&mut part);
        if ci + 1 < chunks.len() {
            pcm.extend_from_slice(pause);
        }
    }
    let out_path = audio::chapter_wav_path(chapters_dir, chapter_index + 1, &ch.title);
    audio::write_wav_mono_i16(&out_path, sample_rate, &pcm)?;
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_exists_and_nonempty_should_work() -> anyhow::Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "epub2tts_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir)?;
        let p = dir.join("a.bin");
        std::fs::write(&p, b"1")?;
        assert!(file_exists_and_nonempty(&p.to_string_lossy()));
        Ok(())
    }
}
