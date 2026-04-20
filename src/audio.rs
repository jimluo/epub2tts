//! 音频工具：把合成得到的 PCM 数据写入 WAV，并生成静音片段用于拼接。

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn sanitize_filename(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == ' ' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim().trim_matches('.').to_string();
    if trimmed.is_empty() || trimmed.chars().all(|c| c == '_' || c == ' ') {
        "chapter".to_string()
    } else {
        trimmed
    }
}

pub fn chapter_wav_path(out_dir: &Path, index_1based: usize, title: &str) -> PathBuf {
    let title = sanitize_filename(title);
    out_dir.join(format!("{index_1based:03}_{title}.wav"))
}

pub fn silence_i16(sample_rate: u32, millis: u32) -> Vec<i16> {
    let samples = (sample_rate as u64 * millis as u64 / 1000) as usize;
    vec![0i16; samples]
}

pub fn write_wav_mono_i16(path: &Path, sample_rate: u32, pcm: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).with_context(|| format!("create {path:?}"))?;
    for s in pcm {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_filename_should_strip_bad_chars() {
        assert_eq!(sanitize_filename("章:1/2"), "章_1_2");
        assert_eq!(sanitize_filename(".."), "chapter");
    }

    #[test]
    fn silence_should_have_expected_length() {
        let pcm = silence_i16(1000, 250);
        assert_eq!(pcm.len(), 250);
    }
}
