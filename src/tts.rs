//! TTS 引擎抽象：统一“文本 -> PCM 音频”的接口，并提供一个无模型依赖的 Fake 引擎用于开发/测试。

use anyhow::Result;
#[cfg(feature = "fish")]
use candle_core::{DType, Device};
#[cfg(feature = "fish")]
use fish_speech_core::{
    config::WhichModel,
    text::{clean::preprocess_text, prompt::PromptEncoder},
};
#[cfg(feature = "fish")]
use server::{
    handlers::speech::{server_lm_generate_blocking, vocode_semantic_tokens},
    state::AppState,
    utils::load::{Args as FishLoadArgs, load_codec, load_lm},
};
#[cfg(feature = "fish")]
use std::{path::Path, sync::Arc};

pub trait TtsEngine {
    fn sample_rate(&self) -> u32;
    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>>;
}

#[cfg(any(feature = "fish", feature = "kokoro", test))]
fn pcm_f32_to_i16(pcm: &[f32], sample_rate: u32) -> Vec<i16> {
    if pcm.is_empty() {
        return Vec::new();
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &x in pcm {
        if x.is_finite() {
            sum += x;
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f32 } else { 0.0 };

    let mut peak = 0.0f32;
    for &x in pcm {
        let y = if x.is_finite() { x - mean } else { 0.0 };
        peak = peak.max(y.abs());
    }

    let max_gain = 3.0f32;
    let gain = if peak > 0.0 {
        (0.98 / peak).min(max_gain)
    } else {
        1.0
    };

    let fade_len = ((sample_rate as usize).max(1) / 100).min(pcm.len() / 2);
    let denom = fade_len.max(1) as f32;

    let mut out = Vec::with_capacity(pcm.len());
    for (i, &x) in pcm.iter().enumerate() {
        let mut y = if x.is_finite() { x - mean } else { 0.0 };
        y *= gain;
        if i < fade_len {
            y *= i as f32 / denom;
        } else if pcm.len() - i <= fade_len {
            y *= (pcm.len() - i) as f32 / denom;
        }
        let s = (y * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32);
        out.push(s as i16);
    }
    out
}

#[cfg(test)]
pub struct FakeEngine {
    sample_rate: u32,
}

#[cfg(test)]
impl FakeEngine {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

#[cfg(test)]
impl TtsEngine for FakeEngine {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>> {
        let len = text.chars().count().max(1) as u32;
        let millis = 20 * len.min(200);
        let total = (self.sample_rate as u64 * millis as u64 / 1000) as usize;
        let mut pcm = Vec::with_capacity(total);
        let freq_hz = 440.0;
        for i in 0..total {
            let t = i as f32 / self.sample_rate as f32;
            let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            let amp = 0.15;
            pcm.push(v * amp);
        }
        Ok(pcm_f32_to_i16(&pcm, self.sample_rate))
    }
}

#[cfg(feature = "kokoro")]
type KokoroVoices = std::collections::HashMap<String, Vec<Vec<Vec<f32>>>>;

#[cfg(feature = "kokoro")]
pub struct KokoroEngine {
    sample_rate: u32,
    session: ort::session::Session,
    voices: KokoroVoices,
    voice_key: String,
    model_is_v11: bool,
    speed_v10: Option<f32>,
    speed_v11: Option<i32>,
}

#[cfg(feature = "kokoro")]
impl KokoroEngine {
    pub fn from_dir(dir: &std::path::Path, voice_name: &str) -> Result<Self> {
        let model_path = dir.join("model.onnx");
        let voices_path = dir.join("voices.bin");

        if !model_path.exists() {
            anyhow::bail!(
                "kokoro model not found at {:?}. Please place model.onnx in the model directory.",
                model_path
            );
        }
        if !voices_path.exists() {
            anyhow::bail!(
                "kokoro voices not found at {:?}. Please place voices.bin in the model directory.",
                voices_path
            );
        }

        let bytes = std::fs::read(&voices_path)?;
        let (voices, _): (KokoroVoices, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard())?;

        let session = ort::session::Session::builder()?.commit_from_file(&model_path)?;
        let model_is_v11 = session.inputs.iter().any(|i| i.name == "input_ids");

        let (voice_key, speed_v10, speed_v11) = parse_kokoro_voice(voice_name, model_is_v11);
        Ok(Self {
            sample_rate: 24_000,
            session,
            voices,
            voice_key,
            model_is_v11,
            speed_v10,
            speed_v11,
        })
    }

    #[allow(dead_code)]
    pub fn from_paths(
        model_path: &std::path::Path,
        voices_path: &std::path::Path,
        voice_name: &str,
        preset: &str,
    ) -> Result<Self> {
        let bytes = std::fs::read(voices_path)?;
        let (voices, _): (KokoroVoices, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard())?;

        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        let model_is_v11 = session.inputs.iter().any(|i| i.name == "input_ids");

        let (voice_key, speed_v10, speed_v11) =
            parse_kokoro_voice_with_preset(voice_name, preset, model_is_v11);
        Ok(Self {
            sample_rate: 24_000,
            session,
            voices,
            voice_key,
            model_is_v11,
            speed_v10,
            speed_v11,
        })
    }
}

#[cfg(feature = "kokoro")]
fn parse_kokoro_voice(name: &str, model_is_v11: bool) -> (String, Option<f32>, Option<i32>) {
    let (base, speed_override) = name.split_once('@').unwrap_or((name, ""));
    let base = base
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "");
    if model_is_v11 {
        let speed = if speed_override.is_empty() {
            1
        } else {
            speed_override.parse::<i32>().unwrap_or(1)
        };
        let key = if base.is_empty() || base == "auto" {
            "zf_048".to_string()
        } else {
            base
        };
        (key, None, Some(speed))
    } else {
        let speed = if speed_override.is_empty() {
            1.0
        } else {
            speed_override.parse::<f32>().unwrap_or(1.0)
        };
        let key = if base.is_empty() || base == "auto" {
            "zf_xiaoxiao".to_string()
        } else {
            base
        };
        (key, Some(speed), None)
    }
}

#[cfg(feature = "kokoro")]
#[allow(dead_code)]
fn parse_kokoro_voice_with_preset(
    name: &str,
    preset: &str,
    model_is_v11: bool,
) -> (String, Option<f32>, Option<i32>) {
    let (base, speed_override) = name.split_once('@').unwrap_or((name, ""));
    let base = base
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "");
    if model_is_v11 {
        let speed = if speed_override.is_empty() {
            if preset == "speed" { 2 } else { 1 }
        } else {
            speed_override
                .parse::<i32>()
                .unwrap_or(if preset == "speed" { 2 } else { 1 })
        };
        let key = if base.is_empty() || base == "auto" {
            "zf_048".to_string()
        } else {
            base
        };
        (key, None, Some(speed))
    } else {
        let speed = if speed_override.is_empty() {
            if preset == "speed" { 1.2 } else { 1.0 }
        } else {
            speed_override
                .parse::<f32>()
                .unwrap_or(if preset == "speed" { 1.2 } else { 1.0 })
        };
        let key = if base.is_empty() || base == "auto" {
            "zf_xiaoxiao".to_string()
        } else {
            base
        };
        (key, Some(speed), None)
    }
}

#[cfg(feature = "kokoro")]
impl TtsEngine for KokoroEngine {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>> {
        let pack = self
            .voices
            .get(&self.voice_key)
            .ok_or_else(|| anyhow::anyhow!("kokoro voice not found: {}", self.voice_key))?;
        let samples = if self.model_is_v11 {
            let speed = self.speed_v11.unwrap_or(1);
            kokoro_synth_v11(&self.session, text, pack.as_slice(), speed)?
        } else {
            let speed = self.speed_v10.unwrap_or(1.0);
            kokoro_synth_v10(&self.session, text, pack.as_slice(), speed)?
        };
        Ok(pcm_f32_to_i16(&samples, self.sample_rate))
    }
}

#[cfg(feature = "kokoro")]
fn kokoro_synth_v10(
    session: &ort::session::Session,
    text: &str,
    pack: &[Vec<Vec<f32>>],
    speed: f32,
) -> Result<Vec<f32>> {
    let phonemes = std::panic::catch_unwind(|| kokoro_tts::g2p(text, false))
        .map_err(|_| anyhow::anyhow!("kokoro g2p panicked"))?
        .map_err(|e| anyhow::anyhow!(e))?;
    let token_ids = kokoro_tts::get_token_ids(&phonemes, false);
    let token_len = token_ids.len();
    let tokens = ndarray::Array::from_shape_vec((1, token_len), token_ids)?;
    let pack_idx = token_len
        .saturating_sub(1)
        .min(pack.len().saturating_sub(1));
    let ref_s = pack
        .get(pack_idx)
        .and_then(|x| x.first())
        .cloned()
        .unwrap_or_default();
    let style = ndarray::Array::from_shape_vec((1, ref_s.len()), ref_s)?;
    let speed_arr = ndarray::Array::from_vec(vec![speed]);

    let out = session.run(ort::inputs![
        "tokens" => tokens.view(),
        "style" => style.view(),
        "speed" => speed_arr.view(),
    ]?)?;
    let audio_t = out["audio"].try_extract_tensor::<f32>()?;
    let audio = audio_t.as_slice();
    audio
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow::anyhow!("kokoro v1.0 model returned empty audio"))
}

#[cfg(feature = "kokoro")]
fn kokoro_synth_v11(
    session: &ort::session::Session,
    text: &str,
    pack: &[Vec<Vec<f32>>],
    speed: i32,
) -> Result<Vec<f32>> {
    let phonemes = std::panic::catch_unwind(|| kokoro_tts::g2p(text, true))
        .map_err(|_| anyhow::anyhow!("kokoro g2p panicked"))?
        .map_err(|e| anyhow::anyhow!(e))?;
    let token_ids = kokoro_tts::get_token_ids(&phonemes, true);
    let token_len = token_ids.len();
    let input_ids = ndarray::Array::from_shape_vec((1, token_len), token_ids)?;
    let pack_idx = token_len
        .saturating_sub(1)
        .min(pack.len().saturating_sub(1));
    let ref_s = pack
        .get(pack_idx)
        .and_then(|x| x.first())
        .cloned()
        .unwrap_or_default();
    let style = ndarray::Array::from_shape_vec((1, ref_s.len()), ref_s)?;
    let speed_arr = ndarray::Array::from_vec(vec![speed]);

    let out = session.run(ort::inputs![
        "input_ids" => input_ids.view(),
        "style" => style.view(),
        "speed" => speed_arr.view(),
    ]?)?;
    let audio_t = out["waveform"].try_extract_tensor::<f32>()?;
    let audio = audio_t.as_slice();
    audio
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow::anyhow!("kokoro v1.1 model returned empty waveform"))
}

#[cfg(feature = "fish")]
pub struct FishSpeechEngine {
    sample_rate: u32,
    voice: String,
    runtime: tokio::runtime::Runtime,
    state: Arc<AppState>,
}

#[cfg(feature = "fish")]
#[derive(Clone)]
pub struct FishSpeechShared {
    sample_rate: u32,
    state: Arc<AppState>,
}

#[cfg(feature = "fish")]
impl FishSpeechEngine {
    pub fn from_dir(model_dir: &Path, voice: &str) -> Result<Self> {
        let shared = FishSpeechShared::from_dir(model_dir, voice)?;
        Self::from_shared(shared, voice)
    }

    pub fn from_shared(shared: FishSpeechShared, voice: &str) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            sample_rate: shared.sample_rate,
            voice: voice.to_string(),
            runtime,
            state: shared.state,
        })
    }
}

#[cfg(feature = "fish")]
impl FishSpeechShared {
    pub fn from_dir(model_dir: &Path, voice: &str) -> Result<Self> {
        if !model_dir.exists() {
            anyhow::bail!(
                "fish model directory not found at {:?}.\n\n\
                Please download Fish Speech model files and place them in this directory.\n\
                Expected files:\n\
                - model.safetensors (or .bin)\n\
                - config.json\n\
                - tokenizer.json\n\
                - firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors\n\
                - voices-template/ directory with voice files",
                model_dir
            );
        }

        // Check for essential model files
        let has_model =
            model_dir.join("model.safetensors").exists() || model_dir.join("model.bin").exists();
        let has_config = model_dir.join("config.json").exists();
        let has_tokenizer = model_dir.join("tokenizer.json").exists();
        let has_codec = model_dir
            .join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors")
            .exists();

        let mut missing = Vec::new();
        if !has_model {
            missing.push("model.safetensors (or model.bin)");
        }
        if !has_config {
            missing.push("config.json");
        }
        if !has_tokenizer {
            missing.push("tokenizer.json");
        }
        if !has_codec {
            missing.push("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors");
        }

        if !missing.is_empty() {
            anyhow::bail!(
                "fish model directory {:?} is incomplete.\n\
                Missing files:\n\
                {}\n\n\
                Please download the complete Fish Speech model files.",
                model_dir,
                missing
                    .iter()
                    .map(|f| format!("  - {}", f))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }

        let checkpoint = model_dir.canonicalize()?;
        let voice_dir = if voice == "unconditioned" {
            model_dir.join("voices-none")
        } else {
            resolve_voice_dir(model_dir)?
        };
        let args = FishLoadArgs {
            checkpoint: Some(checkpoint.clone()),
            fish_version: WhichModel::Fish1_5,
            voice_dir,
            port: 0,
            temp: 0.7,
            top_p: 0.8,
        };
        let device = Device::Cpu;
        let dtype = DType::F32;
        let lm_state = load_lm(&args, Some(checkpoint.clone()), dtype, &device)?;
        let (codec_state, sample_rate) =
            load_codec(&args, dtype, &device, lm_state.config.num_codebooks)?;
        let state = Arc::new(AppState {
            lm: Arc::new(lm_state),
            codec: Arc::new(codec_state),
            model_type: args.fish_version,
            device,
            sample_rate,
        });
        Ok(Self { sample_rate, state })
    }
}

#[cfg(feature = "fish")]
fn resolve_voice_dir(model_dir: &Path) -> Result<std::path::PathBuf> {
    let candidates = [
        model_dir.join("voices"),
        model_dir.join("voices-template"),
        model_dir.to_path_buf(),
    ];
    for dir in candidates {
        if dir.join("index.json").exists() {
            return Ok(dir);
        }
    }
    anyhow::bail!(
        "fish voices not found under {:?}. expected one of: voices/index.json, voices-template/index.json, or index.json in root",
        model_dir
    );
}

#[cfg(feature = "fish")]
impl TtsEngine for FishSpeechEngine {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>> {
        let state = Arc::clone(&self.state);
        let voice = self.voice.clone();
        let audio = self.runtime.block_on(async move {
            let voice_embedding = match voice.as_str() {
                "unconditioned" => None,
                _ => Some(
                    state
                        .lm
                        .voices
                        .lock()
                        .await
                        .get(&voice)
                        .unwrap_or(&state.lm.default_voice)
                        .clone(),
                ),
            };
            let chunks = preprocess_text(text);
            let prompt_encoder = PromptEncoder::new(
                &state.lm.tokenizer,
                &state.device,
                state.lm.config.num_codebooks,
                state.lm.model_type,
            );
            let sysprompt = match state.model_type {
                WhichModel::Fish1_5 => Some("请使用与文本一致的语言自然朗读以下内容。".to_string()),
                _ => None,
            };
            let (n_conditioning_tokens, prompts) =
                prompt_encoder.encode_sequence(chunks, sysprompt, voice_embedding, true)?;
            let mut pcm = Vec::<f32>::new();
            for prompt in &prompts {
                let (semantic_tokens, _) = server_lm_generate_blocking(
                    Arc::clone(&state),
                    prompt,
                    &state.lm.default_sampling_args,
                    n_conditioning_tokens,
                    false,
                )
                .await?;
                let mut part = vocode_semantic_tokens(Arc::clone(&state), &semantic_tokens)
                    .await?
                    .to_vec1::<f32>()?;
                pcm.append(&mut part);
            }
            state.lm.model.lock().await.clear_slow_layer_caches();
            Ok::<Vec<f32>, anyhow::Error>(pcm)
        })?;
        Ok(pcm_f32_to_i16(&audio, self.sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_engine_should_return_audio() -> Result<()> {
        let mut e = FakeEngine::new(10_000);
        let pcm = e.synthesize("你好")?;
        assert!(!pcm.is_empty());
        Ok(())
    }
}
