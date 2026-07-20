import base64
import os

import numpy as np

# 요청된 encoding_format별로 ONNX 그래프에서 받아올 출력.
#
# base64_int8 / base64_binary는 그래프가 직접 내는 공식 출력을 그대로 쓴다 — 우리가
# tanh·반올림·부호판정을 재구현하면 export가 바뀌었을 때 조용히 어긋날 수 있다.
# float / base64는 그래프에 없는 무손실 표현이라 pooler_output(tanh 적용 *전* 값)에서
# 만든다.
#
# 출력을 전부(run(None)) 받지 않는 이유: last_hidden_state가 batch×seq×1024라
# batch 8 / seq 321에서도 10.5MB, batch 32 / seq 2048이면 268MB를 매 요청 복사한다.
# 지연 차이는 없지만(연산은 어차피 수행됨) 메모리 스파이크가 크다.
OUTPUT_FOR = {
    "float": "pooler_output",
    "base64_float32": "pooler_output",
    "base64_int8": "pooler_output_int8",
    "base64_binary": "pooler_output_binary",
}

# base64는 공식 enum에 없지만 OpenAI SDK가 생략 시 자동으로 붙이는 값이다.
# 공식 기본값(base64_int8)과 같은 의미를 주도록 별칭 처리한다.
ALIASES = {"base64": "base64_int8"}

DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/models/perplexity-ai-pplx-embed-v1-0.6b-int8"
)
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))

# schemas.EmbeddingRequest.encoding_format의 기본값과 반드시 같아야 한다
# (테스트 test_embedder_default_matches_schema_default가 지킨다).
DEFAULT_ENCODING_FORMAT = "base64_int8"


def cgroup_cpu_limit(root: str = "/sys/fs/cgroup"):
    """컨테이너에 걸린 CPU limit(코어 수)을 읽는다. 무제한이면 None.

    cgroup v2는 `cpu.max`("<quota|max> <period>"), v1은 cpu.cfs_quota_us /
    cpu.cfs_period_us. 둘 다 없거나 무제한이면 None.
    """
    try:
        with open(f"{root}/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass

    try:
        with open(f"{root}/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open(f"{root}/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


def effective_cpu_count(root: str = "/sys/fs/cgroup") -> int:
    """이 프로세스가 실제로 쓸 수 있는 코어 수.

    onnxruntime은 기본적으로 호스트 전체 코어 수만큼 intra-op 스레드를 띄운다.
    cgroup limit을 무시하므로, 코어 많은 노드에서는 CFS 스로틀링과 컨텍스트
    스위칭 때문에 오히려 느려진다. 그래서 limit을 직접 읽어 넘긴다.
    """
    host = os.cpu_count() or 1
    limit = cgroup_cpu_limit(root)
    if limit is None:
        return host
    return max(1, min(host, int(limit)))


def canonical(encoding_format: str) -> str:
    """별칭을 공식 이름으로 바꾼다."""
    return ALIASES.get(encoding_format, encoding_format)


def output_name(encoding_format: str) -> str:
    """요청된 인코딩을 만들기 위해 그래프에서 받아야 할 출력 이름."""
    try:
        return OUTPUT_FOR[canonical(encoding_format)]
    except KeyError:
        raise ValueError(f"Invalid encoding_format: {encoding_format}") from None


def truncate(raw: np.ndarray, dimensions):
    """Matryoshka 차원 축소.

    tanh / 반올림 / 부호판정은 모두 elementwise라 슬라이싱과 교환 가능하다.
    그래서 그래프 출력을 그대로 자르면 되고, 잘라낸 뒤 재정규화만 하면 된다
    (float 경로의 정규화는 postprocess에서 수행).
    base64_binary는 packbits 전 부호 배열이라 여기서 자르는 것이 맞다.
    """
    if dimensions is None:
        return raw
    return raw[:, :dimensions]


def postprocess(raw: np.ndarray, encoding_format: str):
    """그래프 출력을 최종 응답 값으로 변환한다.

    float는 tanh 후 L2 정규화 — 정규화는 벡터당 스칼라 나눗셈이라 방향(=코사인)을
    바꾸지 않으므로, int8과 같은 공간에 있으면서 반올림 손실만 없앤 값이 된다.
    base64_binary는 공식 PackedBinaryQuantizer와 같다: binary 출력이 x>=0에서
    +1이므로 packbits(binary > 0) == packbits(x >= 0).
    """
    encoding_format = canonical(encoding_format)

    if encoding_format in ("float", "base64_float32"):
        v = np.tanh(raw).astype(np.float32)
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        # 전부 0인 벡터에서 0으로 나누지 않도록.
        v = v / np.where(norm == 0, 1.0, norm)
        if encoding_format == "float":
            return [row.tolist() for row in v]
        return [base64.b64encode(row.tobytes()).decode() for row in v]

    if encoding_format == "base64_int8":
        return [base64.b64encode(row.tobytes()).decode() for row in raw]

    if encoding_format == "base64_binary":
        packed = np.packbits(raw > 0, axis=-1)
        return [base64.b64encode(row.tobytes()).decode() for row in packed]

    raise ValueError(f"Invalid encoding_format: {encoding_format}")


class PplxEmbedder:
    """ONNX Runtime 세션 하나를 감싼 임베더. 프로세스당 한 인스턴스."""

    def __init__(self, session, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self._session = session
        self._tokenizer = tokenizer
        tokenizer.enable_truncation(max_length)
        tokenizer.enable_padding()

    @classmethod
    def load(
        cls,
        model_dir: str = DEFAULT_MODEL_DIR,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> "PplxEmbedder":
        # 지연 임포트 — 테스트가 실제 런타임 없이 이 모듈을 임포트할 수 있게.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()

        # 스레드 수: 환경변수 > cgroup limit > 호스트 코어 수.
        threads = int(os.environ.get("ORT_INTRA_OP_THREADS") or effective_cpu_count())
        opts.intra_op_num_threads = threads
        # 그래프가 하나의 직렬 체인이라 연산자 병렬은 이득이 없다. 1로 고정해
        # 불필요한 스레드 풀을 만들지 않는다.
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 기본값이면 워커 스레드가 다음 요청을 스핀 대기하며 CPU를 태운다.
        # 요청이 드문드문 오는 서버라 idle 낭비가 커서 끈다(대신 wake 지연 약간).
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")

        # 입력 길이가 요청마다 달라 메모리 패턴 예측이 맞지 않는다. 켜 두면
        # 형상마다 패턴을 잡아 두느라 메모리만 늘어난다.
        opts.enable_mem_pattern = False

        session = ort.InferenceSession(
            f"{model_dir}/onnx/model.onnx",
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        print(
            f"[pplx-embed] loaded {model_dir} "
            f"intra_op_threads={threads} (host={os.cpu_count()}, "
            f"cgroup_limit={cgroup_cpu_limit()}) max_length={max_length}",
            flush=True,
        )
        return cls(session=session, tokenizer=tokenizer, max_length=max_length)

    def embed(
        self,
        texts: list[str],
        encoding_format: str = DEFAULT_ENCODING_FORMAT,
        dimensions=None,
    ):
        """(인코딩된 임베딩 리스트, 패딩 제외 토큰 수)를 돌려준다."""
        name = output_name(encoding_format)

        encs = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        (raw,) = self._session.run([name], {"input_ids": ids, "attention_mask": mask})
        return postprocess(truncate(raw, dimensions), encoding_format), int(mask.sum())
