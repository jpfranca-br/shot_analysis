from pathlib import Path
import threading
from typing import Dict, List, Optional, Tuple
import urllib.request

import h5py
import librosa
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

YAMNET_H5_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
YAMNET_SR = 16000

# AudioSet Class Indices in Google YAMNet (521 classes)
GUNSHOT_CLASS_INDICES = [
    420,  # Explosion
    421,  # Gunshot, gunfire
    422,  # Machine gun
    423,  # Fusillade
    424,  # Artillery fire
    425,  # Cap gun
    426,  # Fireworks
    427,  # Firecracker
    428,  # Burst, pop
    429,  # Eruption
    430,  # Boom
    434,  # Crack
    161,  # Rimshot / sharp percussive snap
]

BEEP_CLASS_INDICES = [
    475,  # Beep, bleep
    382,  # Alarm
    389,  # Alarm clock
    392,  # Buzzer
    393,  # Smoke detector, smoke alarm
    394,  # Fire alarm
    313,  # Reversing beeps
    495,  # Sine wave
    396,  # Whistle
]

SPEECH_CLASS_INDICES = [
    0,    # Speech
    1,    # Male speech
    2,    # Female speech
    3,    # Child speech
    4,    # Conversation
    5,    # Narration, monologue
]

# Official YAMNet MobileNetV1 layer definitions: (layer_name, stride, out_channels)
YAMNET_LAYER_SPECS = [
    ("layer1", 2, 32),
    ("layer2", 1, 64),
    ("layer3", 2, 128),
    ("layer4", 1, 128),
    ("layer5", 2, 256),
    ("layer6", 1, 256),
    ("layer7", 2, 512),
    ("layer8", 1, 512),
    ("layer9", 1, 512),
    ("layer10", 1, 512),
    ("layer11", 1, 512),
    ("layer12", 1, 512),
    ("layer13", 2, 1024),
    ("layer14", 1, 1024),
]


def _fuse_conv_bn(
    weight_oihw: np.ndarray,
    beta: np.ndarray,
    moving_mean: np.ndarray,
    moving_var: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse Conv2D weights (O, I, H, W) with BatchNorm (gamma=1.0, beta, mean, var, eps=1e-4).
    """
    scale = 1.0 / np.sqrt(moving_var + eps)
    w_fused = weight_oihw * scale.reshape(-1, 1, 1, 1)
    b_fused = beta - moving_mean * scale
    return w_fused.astype(np.float32), b_fused.astype(np.float32)


def build_yamnet_onnx(h5_path: Path, onnx_path: Path) -> Path:
    """
    Convert official Google YAMNet Keras HDF5 weights (yamnet.h5) into an optimized
    fused ONNX graph (yamnet.onnx) outputting both:
      - `scores`: [N, 521] AudioSet class probabilities
      - `embeddings`: [N, 1024] penultimate neural embedding vectors
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    initializers = []

    def add_init(name: str, arr: np.ndarray) -> str:
        tensor = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=list(arr.shape),
            vals=arr.flatten().tolist(),
        )
        initializers.append(tensor)
        return name

    with h5py.File(h5_path, "r") as f:
        # Layer 1: Standard Conv 3x3, stride 2
        k1 = np.array(f["layer1/conv/layer1/conv/kernel:0"], dtype=np.float32)  # (3, 3, 1, 32) HWIO
        k1_oihw = np.transpose(k1, (3, 2, 0, 1))  # (32, 1, 3, 3) OIHW
        beta1 = np.array(f["layer1/conv/bn/layer1/conv/bn/beta:0"], dtype=np.float32)
        mean1 = np.array(f["layer1/conv/bn/layer1/conv/bn/moving_mean:0"], dtype=np.float32)
        var1 = np.array(f["layer1/conv/bn/layer1/conv/bn/moving_variance:0"], dtype=np.float32)
        w1, b1 = _fuse_conv_bn(k1_oihw, beta1, mean1, var1)

        add_init("layer1_w", w1)
        add_init("layer1_b", b1)
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=["mel_patches", "layer1_w", "layer1_b"],
                outputs=["layer1_conv_out"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_UPPER",
            )
        )
        nodes.append(
            helper.make_node("Relu", inputs=["layer1_conv_out"], outputs=["layer1_out"])
        )
        prev_out = "layer1_out"
        in_channels = 32

        # Layers 2..14: Depthwise Separable Conv blocks
        for layer_name, stride, out_channels in YAMNET_LAYER_SPECS[1:]:
            # Depthwise Conv 3x3
            dw_k = np.array(
                f[f"{layer_name}/depthwise_conv/{layer_name}/depthwise_conv/depthwise_kernel:0"],
                dtype=np.float32,
            )  # (3, 3, C, 1)
            dw_oihw = np.transpose(dw_k, (2, 3, 0, 1))  # (C, 1, 3, 3)
            dw_beta = np.array(
                f[f"{layer_name}/depthwise_conv/bn/{layer_name}/depthwise_conv/bn/beta:0"],
                dtype=np.float32,
            )
            dw_mean = np.array(
                f[f"{layer_name}/depthwise_conv/bn/{layer_name}/depthwise_conv/bn/moving_mean:0"],
                dtype=np.float32,
            )
            dw_var = np.array(
                f[f"{layer_name}/depthwise_conv/bn/{layer_name}/depthwise_conv/bn/moving_variance:0"],
                dtype=np.float32,
            )
            dw_w, dw_b = _fuse_conv_bn(dw_oihw, dw_beta, dw_mean, dw_var)

            add_init(f"{layer_name}_dw_w", dw_w)
            add_init(f"{layer_name}_dw_b", dw_b)
            dw_pad = "SAME_UPPER" if stride == 2 else "SAME_LOWER"
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=[prev_out, f"{layer_name}_dw_w", f"{layer_name}_dw_b"],
                    outputs=[f"{layer_name}_dw_conv"],
                    kernel_shape=[3, 3],
                    strides=[stride, stride],
                    group=in_channels,
                    auto_pad=dw_pad,
                )
            )
            nodes.append(
                helper.make_node(
                    "Relu",
                    inputs=[f"{layer_name}_dw_conv"],
                    outputs=[f"{layer_name}_dw_relu"],
                )
            )

            # Pointwise Conv 1x1
            pw_k = np.array(
                f[f"{layer_name}/pointwise_conv/{layer_name}/pointwise_conv/kernel:0"],
                dtype=np.float32,
            )  # (1, 1, C_in, C_out)
            pw_oihw = np.transpose(pw_k, (3, 2, 0, 1))  # (C_out, C_in, 1, 1)
            pw_beta = np.array(
                f[f"{layer_name}/pointwise_conv/bn/{layer_name}/pointwise_conv/bn/beta:0"],
                dtype=np.float32,
            )
            pw_mean = np.array(
                f[f"{layer_name}/pointwise_conv/bn/{layer_name}/pointwise_conv/bn/moving_mean:0"],
                dtype=np.float32,
            )
            pw_var = np.array(
                f[f"{layer_name}/pointwise_conv/bn/{layer_name}/pointwise_conv/bn/moving_variance:0"],
                dtype=np.float32,
            )
            pw_w, pw_b = _fuse_conv_bn(pw_oihw, pw_beta, pw_mean, pw_var)

            add_init(f"{layer_name}_pw_w", pw_w)
            add_init(f"{layer_name}_pw_b", pw_b)
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=[
                        f"{layer_name}_dw_relu",
                        f"{layer_name}_pw_w",
                        f"{layer_name}_pw_b",
                    ],
                    outputs=[f"{layer_name}_pw_conv"],
                    kernel_shape=[1, 1],
                    strides=[1, 1],
                )
            )
            out_name = f"{layer_name}_out"
            nodes.append(
                helper.make_node("Relu", inputs=[f"{layer_name}_pw_conv"], outputs=[out_name])
            )
            prev_out = out_name
            in_channels = out_channels

        # Global Average Pool -> Squeeze to [N, 1024] embeddings
        nodes.append(
            helper.make_node(
                "GlobalAveragePool",
                inputs=[prev_out],
                outputs=["pooled_4d"],
            )
        )
        axes_tensor = helper.make_tensor(
            name="squeeze_axes",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[2, 3],
        )
        initializers.append(axes_tensor)
        nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=["pooled_4d", "squeeze_axes"],
                outputs=["embeddings"],
            )
        )

        # Final Dense Classification Layer -> Sigmoid [N, 521]
        fc_w = np.array(f["logits/logits/kernel:0"], dtype=np.float32)  # (1024, 521)
        fc_b = np.array(f["logits/logits/bias:0"], dtype=np.float32)    # (521,)
        add_init("logits_w", fc_w)
        add_init("logits_b", fc_b)
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=["embeddings", "logits_w", "logits_b"],
                outputs=["logits_out"],
            )
        )
        nodes.append(
            helper.make_node("Sigmoid", inputs=["logits_out"], outputs=["scores"])
        )

    input_info = helper.make_tensor_value_info(
        "mel_patches", TensorProto.FLOAT, [None, 1, 96, 64]
    )
    scores_info = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [None, 521]
    )
    emb_info = helper.make_tensor_value_info(
        "embeddings", TensorProto.FLOAT, [None, 1024]
    )

    graph = helper.make_graph(
        nodes,
        "Google_YAMNet_AudioSet",
        inputs=[input_info],
        outputs=[scores_info, emb_info],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="shot_analysis_yamnet",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(onnx_path))
    return onnx_path


class YAMNetClassifier:
    """
    Singleton wrapper around Google's YAMNet running on local ONNX Runtime.
    Provides AudioSet gunshot/beep probabilities and 64-D pooled neural embeddings.
    """

    _instance: Optional["YAMNetClassifier"] = None
    _lock = threading.Lock()

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.models_dir / "yamnet.h5"
        self.onnx_path = self.models_dir / "yamnet.onnx"
        self._session: Optional[ort.InferenceSession] = None
        self._mel_basis: Optional[np.ndarray] = None

    @classmethod
    def get_instance(cls) -> "YAMNetClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def ensure_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session

        with self._lock:
            if self._session is not None:
                return self._session

            if not self.onnx_path.exists():
                if not self.h5_path.exists():
                    urllib.request.urlretrieve(YAMNET_H5_URL, str(self.h5_path))
                build_yamnet_onnx(self.h5_path, self.onnx_path)

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            return self._session

    def _get_mel_basis(self) -> np.ndarray:
        if self._mel_basis is None:
            self._mel_basis = librosa.filters.mel(
                sr=YAMNET_SR,
                n_fft=512,
                n_mels=64,
                fmin=125.0,
                fmax=7500.0,
                htk=True,
                norm=None,
            ).astype(np.float32)
        return self._mel_basis

    def waveform_to_log_mel(self, y_16k: np.ndarray) -> np.ndarray:
        """
        Compute YAMNet-compatible log-mel spectrogram frames (10ms hop, 25ms Hann window, 64 mel bins).
        Returns array of shape [num_frames, 64].
        """
        win_length = int(0.025 * YAMNET_SR)  # 400 samples
        hop_length = int(0.010 * YAMNET_SR)  # 160 samples
        n_fft = 512

        if len(y_16k) < win_length:
            y_16k = np.pad(y_16k, (0, win_length - len(y_16k)))

        S = np.abs(
            librosa.stft(
                y_16k,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=False,
            )
        )
        mel_spec = np.dot(self._get_mel_basis(), S)  # [64, n_frames]
        log_mel = np.log(mel_spec.T + 0.001).astype(np.float32)  # [n_frames, 64]
        return log_mel

    @staticmethod
    def compress_embedding_64d(emb_1024: np.ndarray) -> List[float]:
        """
        Compress a 1024-D YAMNet penultimate embedding into a compact, L2-normalized
        64-D neural signature vector via 16-channel block pooling.
        """
        vec_64 = np.mean(emb_1024.reshape(64, 16), axis=1)
        norm = float(np.linalg.norm(vec_64))
        if norm > 1e-8:
            vec_64 = vec_64 / norm
        return [round(float(x), 6) for x in vec_64]

    def analyze_windows(
        self,
        y: np.ndarray,
        sr: int,
        timestamps: List[float],
        regimes: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        Run batched YAMNet inference on 0.96s patches anchored around each candidate timestamp.
        Supports regime-specific focus windows:
          - "slow" (>2000ms apart or first shot): full 0.55s unmasked ballistic + decay tail
          - "fast" (<=2000ms apart): tight 0.22s transient focus window so adjacent fast shots
            do not bleed into the embedding
        Returns a list of dicts containing:
          - gunshot_prob: float (aggregated probability across AudioSet gunshot/impulse classes)
          - beep_prob: float (max probability across AudioSet beep/alarm/sine classes)
          - speech_prob: float (probability of speech/chatter)
          - embedding_64d: List[float] (64-D L2-normalized YAMNet neural embedding)
        """
        if not timestamps:
            return []

        if sr != YAMNET_SR:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=YAMNET_SR)
        else:
            y_16k = y

        peak = float(np.max(np.abs(y_16k))) if len(y_16k) > 0 else 0.0
        if peak > 1e-6:
            y_16k = (y_16k / peak) * 0.95

        patch_samples = int(0.975 * YAMNET_SR)
        patches = []

        for idx, t in enumerate(timestamps):
            regime = regimes[idx] if (regimes and idx < len(regimes)) else "fast"
            start_sample = int((t - 0.10) * YAMNET_SR)
            end_sample = start_sample + patch_samples

            if start_sample < 0:
                pad_left = -start_sample
                seg = y_16k[0 : max(0, end_sample)]
                seg = np.pad(seg, (pad_left, 0))
            else:
                seg = y_16k[start_sample : min(len(y_16k), end_sample)]

            if len(seg) < patch_samples:
                seg = np.pad(seg, (0, patch_samples - len(seg)))

            # Apply regime-specific temporal attention envelope:
            # Slow shots (>2000ms) capture full 550ms ballistic & room decay signature;
            # Fast shots (<=2000ms) focus tightly on the 220ms direct transient.
            if regime == "slow":
                env_weights = np.ones_like(seg, dtype=np.float32) * 0.45
                focus_start = int(0.06 * YAMNET_SR)
                focus_end = min(len(seg), int(0.65 * YAMNET_SR))
                env_weights[focus_start:focus_end] = 1.0
            else:
                env_weights = np.ones_like(seg, dtype=np.float32) * 0.18
                focus_start = int(0.07 * YAMNET_SR)
                focus_end = min(len(seg), int(0.32 * YAMNET_SR))
                env_weights[focus_start:focus_end] = 1.0

            seg_focused = seg * env_weights

            log_mel = self.waveform_to_log_mel(seg_focused)
            if len(log_mel) < 96:
                log_mel = np.pad(log_mel, ((0, 96 - len(log_mel)), (0, 0)))
            else:
                log_mel = log_mel[:96, :]

            patches.append(log_mel[np.newaxis, :, :])  # [1, 96, 64]

        batch = np.stack(patches, axis=0).astype(np.float32)  # [N, 1, 96, 64]
        session = self.ensure_session()
        scores, embeddings = session.run(
            ["scores", "embeddings"], {"mel_patches": batch}
        )

        results = []
        for idx in range(len(timestamps)):
            sc = scores[idx]
            emb = embeddings[idx]

            gun_probs = sc[GUNSHOT_CLASS_INDICES]
            # Combine top gunshot/impulse class probability with top-3 mean
            top_gun = float(np.max(gun_probs))
            top3_gun = float(np.mean(np.sort(gun_probs)[-3:]))
            gunshot_prob = round(float(0.65 * top_gun + 0.35 * top3_gun), 5)

            beep_prob = round(float(np.max(sc[BEEP_CLASS_INDICES])), 5)
            speech_prob = round(float(np.max(sc[SPEECH_CLASS_INDICES])), 5)

            results.append(
                {
                    "gunshot_prob": gunshot_prob,
                    "beep_prob": beep_prob,
                    "speech_prob": speech_prob,
                    "embedding_64d": self.compress_embedding_64d(emb),
                }
            )

        return results
