import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List, Union, Dict, Any, Iterator
from dataclasses import dataclass
import warnings
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 用於模型輸出的結構
@dataclass
class NagatoSakuraOutput:
    last_hidden_state: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    loss: Optional[torch.FloatTensor] = None

# 對話歷史管理
@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None

class ConversationHistory:
    """對話歷史管理器"""
    
    def __init__(self, max_turns: int = 50, max_length: int = 4096):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        self.max_length = max_length
    
    def add_turn(self, role: str, content: str, timestamp: Optional[float] = None):
        """添加對話輪次"""
        import time
        if timestamp is None:
            timestamp = time.time()
        
        turn = ConversationTurn(role=role, content=content, timestamp=timestamp)
        self.turns.append(turn)
        
        # 保持最大輪次限制
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_conversation_text(self, user_prefix: str = "用戶：", 
                            assistant_prefix: str = "長門櫻：") -> str:
        """獲取格式化的對話文本"""
        formatted_turns = []
        for turn in self.turns:
            if turn.role == "user":
                formatted_turns.append(f"{user_prefix}{turn.content}")
            elif turn.role == "assistant":
                formatted_turns.append(f"{assistant_prefix}{turn.content}")
        
        return "\n".join(formatted_turns)
    
    def clear(self):
        """清空對話歷史"""
        self.turns.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化為字典"""
        return {
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp
                }
                for turn in self.turns
            ],
            "max_turns": self.max_turns,
            "max_length": self.max_length
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """從字典反序列化"""
        history = cls(
            max_turns=data.get("max_turns", 50),
            max_length=data.get("max_length", 4096)
        )
        
        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                role=turn_data["role"],
                content=turn_data["content"],
                timestamp=turn_data.get("timestamp")
            )
            history.turns.append(turn)
        
        return history

# RMSNorm
class NSRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# 旋轉位置編碼
class NSRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 預計算 cos 和 sin 緩存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=device or torch.device('cpu'), 
            dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype)[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype)[None, None, :, :], persistent=False)

    def forward(self, value: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len_in = value.shape[-2]
        if seq_len is None:
            seq_len = seq_len_in

        if seq_len > self.max_seq_len_cached or self.cos_cached.device != value.device or self.cos_cached.dtype != value.dtype:
            self._set_cos_sin_cache(seq_len=max(seq_len, self.max_seq_len_cached), device=value.device, dtype=value.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=value.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=value.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋轉張量的一半維度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """應用旋轉位置編碼"""
    if cos.dim() == 4:  # [1, 1, seq_len, dim]
        cos = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
    if sin.dim() == 4:
        sin = sin.squeeze(0).squeeze(0)
    if cos.shape[0] < q.shape[2]:  # seq_len dimension
        # 如果cos/sin長度不足，只取可用的部分
        q = q[:, :, :cos.shape[0], :]
        k = k[:, :, :cos.shape[0], :]
    elif cos.shape[0] > q.shape[2]:
        # 如果cos/sin過長，截取需要的部分
        cos = cos[:q.shape[2], :]
        sin = sin[:q.shape[2], :]
    
    # 擴展維度以匹配q和k
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 多頭注意力機制
class NSAttention(nn.Module):
    def __init__(self, config: 'NSConfig'):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} 不能被 num_attention_heads {self.num_attention_heads} 整除")

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = NSRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

        self.dropout = nn.Dropout(min(config.attention_dropout, 0.1))
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 投影到 Q, K, V
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        # 重塑為多頭格式
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 計算序列長度
        kv_seq_len = key_states.shape[-2]
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value[0].shape[-2]
            kv_seq_len += past_kv_len

        # 應用 RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # 處理位置 ID
        if position_ids is None:
            position_ids = torch.arange(
                past_kv_len, past_kv_len + q_len, 
                device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)

        # 確保position_ids維度正確
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        # RoPE應用
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except Exception as e:
            # 如果RoPE失敗，繼續使用原始的query和key
            logging.getLogger(__name__).warning(f"RoPE應用失敗，跳過: {e}")

        # 處理過去的鍵值對
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # 使用手動實現的注意力機制
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 使用 softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑輸出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 輸出投影
        attn_output = self.output(attn_output)

        return attn_output, attn_weights if output_attentions else None, present_key_value
    
# SwiGLU MLP
class NSMLP(nn.Module):
    def __init__(self, config: 'NSConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        
        # SwiGLU 激活
        intermediate_output = F.silu(gate_output) * up_output
        
        output = self.down_proj(intermediate_output)
        return output

# 解碼器層
class NSDecoderLayer(nn.Module):
    def __init__(self, config: 'NSConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 注意力機制
        self.self_attn = NSAttention(config)

        # MLP 層
        self.mlp = NSMLP(config)

        # 使用 RMSNorm
        self.input_layernorm = NSRMSNorm(config.hidden_size)
        self.post_attention_layernorm = NSRMSNorm(config.hidden_size)

        # 降低 dropout
        self.dropout = nn.Dropout(min(config.hidden_dropout, 0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        # Pre-Norm 注意力
        hidden_states_norm = self.input_layernorm(hidden_states)
        
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
        present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None

        # 殘差連接
        hidden_states = residual + attn_output

        # Pre-Norm MLP
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        
        # 殘差連接
        hidden_states = residual + self.dropout(mlp_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

# 模型配置
@dataclass
class NSConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 8192
    hidden_dropout: float = 0.05
    attention_dropout: float = 0.05
    memory_tokens: int = 0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    pad_token_id: Optional[int] = 0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    
    # 穩定性參數
    use_stable_embedding: bool = True
    max_grad_norm: float = 1.0
    layer_norm_type: str = "rmsnorm"
    
    # 對話相關配置
    conversation_template: str = "nagato_sakura"  # 對話模板
    system_message: str = "你是長門櫻，一個友善且樂於助人的AI助手。"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NSConfig':
        """從字典創建配置"""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'NSConfig':
        """從文件加載配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# 主模型
class NagatoSakuraModel(nn.Module):
    def __init__(self, config: NSConfig):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        # 詞嵌入層（使用更穩定的初始化）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 解碼器層
        self.layers = nn.ModuleList([NSDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 最終歸一化層
        self.final_layernorm = NSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 梯度檢查點（節省記憶體）
        self.gradient_checkpointing = False

        # 初始化權重
        self.post_init()

    def _init_weights(self, module: nn.Module):
        """改進的權重初始化"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, NSRMSNorm):
            torch.nn.init.ones_(module.weight)

    def post_init(self):
        """應用權重初始化"""
        self.apply(self._init_weights)
        
        # 記錄參數數量
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"模型參數數量: {num_params/1e6:.2f}M")
        logger.info(f"可訓練參數數量: {num_trainable_params/1e6:.2f}M")

    def enable_gradient_checkpointing(self):
        """啟用梯度檢查點以節省記憶體"""
        self.gradient_checkpointing = True

    def _prepare_decoder_attention_mask(
        self, attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], 
        inputs_embeds: torch.Tensor, past_key_values_length: int
    ) -> Optional[torch.Tensor]:
        """準備解碼器注意力掩碼"""
        bsz, seq_len = input_shape
        combined_attention_mask = None

        if seq_len > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=seq_len, 
                past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

            if combined_attention_mask is None:
                combined_attention_mask = expanded_attn_mask
            else:
                combined_attention_mask = torch.minimum(combined_attention_mask, expanded_attn_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, NagatoSakuraOutput]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', True)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同時指定 input_ids 和 inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("必須指定 input_ids 或 inputs_embeds")

        # 處理過去的鍵值對
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]

        # 處理位置 ID
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, 
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # 嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 準備注意力掩碼
        attention_mask_for_layers = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # 初始化輸出收集器
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        # 通過解碼器層
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 安全獲取past_key_value
            past_key_value = None
            if past_key_values is not None and idx < len(past_key_values):
                past_key_value = past_key_values[idx]

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask_for_layers,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_for_layers,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # 安全地處理cache
            if use_cache and len(layer_outputs) > 2 and layer_outputs[2] is not None:
                if next_decoder_cache is None:
                    next_decoder_cache = []
                next_decoder_cache.append(layer_outputs[2])

            if output_attentions and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                all_self_attns += (layer_outputs[1],)

        # 最終歸一化
        hidden_states = self.final_layernorm(hidden_states)

        # 添加最後的隱藏狀態
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache and next_decoder_cache else None

        return NagatoSakuraOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            logits=None,
            loss=None,
        )

# 因果語言模型
class NagatoSakuraForCausalLM(nn.Module):
    def __init__(self, config: NSConfig):
        super().__init__()
        self.config = config
        self.model = NagatoSakuraModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化權重
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_lm_head: nn.Linear):
        self.lm_head = new_lm_head

    def set_decoder(self, decoder: NagatoSakuraModel):
        self.model = decoder

    def get_decoder(self) -> NagatoSakuraModel:
        return self.model

    def _init_weights(self, module: nn.Module):
        """權重初始化"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def post_init(self):
        """應用權重初始化和綁定權重"""
        self.apply(self._init_weights)
        self.tie_weights()

    def tie_weights(self):
        """綁定輸入輸出嵌入權重"""
        if getattr(self.config, "tie_word_embeddings", False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()

            if output_embeddings is not None and input_embeddings is not None:
                if output_embeddings.weight.shape == input_embeddings.weight.shape:
                    output_embeddings.weight = input_embeddings.weight
                    logger.info("詞嵌入權重已綁定")
                else:
                    logger.warning("詞嵌入權重形狀不匹配，無法綁定")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, NagatoSakuraOutput]:

        outputs: NagatoSakuraOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # 移位標籤以進行語言建模
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 計算損失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        outputs.logits = logits
        outputs.loss = loss

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
        attention_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, **kwargs
    ) -> Dict[str, Any]:
        """為生成準備輸入"""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], beam_idx: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """重新排序緩存以支持束搜索"""
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            )
        return reordered_past

    def _apply_generation_config(self, next_token_logits: torch.Tensor, 
                                 generated_tokens: torch.Tensor,
                                 temperature: float = 1.0, 
                                 top_k: int = 50, 
                                 top_p: float = 1.0,
                                 repetition_penalty: float = 1.0, 
                                 do_sample: bool = True) -> torch.Tensor:
        """使用與舊版本完全相同的生成邏輯"""
        
        # 應用重複懲罰 - 只對已生成的token
        if repetition_penalty != 1.0 and generated_tokens.shape[1] > 0:
            # 獲取已生成token的分數
            score = torch.gather(next_token_logits, 1, generated_tokens)
            # 應用懲罰
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            # 將懲罰後的分數放回
            next_token_logits.scatter_(1, generated_tokens, score)

        # 貪婪解碼或採樣
        if do_sample:
            # 溫度調節
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k 過濾
            if top_k > 0:
                top_k_actual = min(top_k, next_token_logits.size(-1))
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k_actual)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)
            
            # Top-p 過濾
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 確保至少有一個有效選項
            if torch.all(torch.isinf(next_token_logits)):
                # 重置為原始logits
                next_token_logits = self.lm_head(self.model.final_layernorm(
                    self.model.embed_tokens(generated_tokens[:, -1:])
                ))[:, -1, :]
            
            # 採樣
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # 貪婪解碼
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        return next_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = 20,
        min_length: Optional[int] = 0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> torch.LongTensor:
        """使用舊版本的生成邏輯確保一致性"""
        if input_ids is None:
            raise ValueError("必須提供 input_ids")

        # 確保模型處於評估模式
        self.eval()

        # 獲取配置中的默認值
        pad_token_id = pad_token_id or getattr(self.config, 'pad_token_id', None)
        eos_token_id = eos_token_id or getattr(self.config, 'eos_token_id', None)

        batch_size, current_length = input_ids.shape
        device = input_ids.device
        
        # 確保max_length大於當前長度
        max_length = max(max_length, current_length + 1)
        
        # 初始化past_key_values為None
        past_key_values = None
        
        # 確保注意力掩碼覆蓋初始輸入
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 生成循環
        for step in range(max_length - current_length):
            try:
                # 準備模型輸入
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                }
                
                # 如果有past_key_values，只使用最後一個token
                if past_key_values is not None:
                    model_inputs["input_ids"] = input_ids[:, -1:]

                outputs: NagatoSakuraOutput = self(**model_inputs)

                if outputs.logits is None:
                    break
                
                next_token_logits = outputs.logits[:, -1, :]

                # 使用舊版本的生成邏輯
                next_tokens = self._apply_generation_config(
                    next_token_logits, input_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample
                )

                # 添加新令牌到序列
                next_tokens = next_tokens.unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                current_length += 1

                # 更新注意力掩碼
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=-1)
                
                # 更新過去的鍵值對
                past_key_values = outputs.past_key_values

                # 檢查停止條件
                if eos_token_id is not None and (next_tokens.squeeze(-1) == eos_token_id).any():
                    break
                    
            except Exception as e:
                # 如果生成過程中出錯，至少返回當前結果
                logging.getLogger(__name__).warning(f"生成過程中出錯，提前結束: {e}")
                break

        # 處理最小長度
        if current_length < min_length and pad_token_id is not None:
            num_padding = min_length - current_length
            padding = torch.full(
                (batch_size, num_padding), pad_token_id, 
                dtype=torch.long, device=device
            )
            input_ids = torch.cat([input_ids, padding], dim=1)

        return input_ids

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        流式生成，使用與generate()完全相同的邏輯
        """
        if input_ids is None:
            raise ValueError("必須提供 input_ids")
        
        # 確保模型處於評估模式
        self.eval()
        
        # 初始化
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # 獲取特殊token
        pad_token_id = getattr(self.config, 'pad_token_id', None)
        eos_token_id = getattr(self.config, 'eos_token_id', None)
        
        # 初始化注意力掩碼
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 初始化生成狀態
        generated_ids = input_ids.clone()
        past_key_values = None
        tokens_generated = 0
        
        # 初始狀態輸出
        yield {
            "token_id": None,
            "token_text": "",
            "generated_text": "",
            "finished": False,
            "stop_reason": None,
            "tokens_generated": 0,
            "input_length": input_length
        }
        
        # 生成循環
        for step in range(max_new_tokens):
            try:
                # 準備輸入
                model_inputs = {
                    "input_ids": generated_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                
                # 如果有past_key_values，只使用最後一個token
                if past_key_values is not None:
                    model_inputs["input_ids"] = generated_ids[:, -1:]
                
                # 前向傳播
                with torch.no_grad():
                    outputs = self(**model_inputs)
                
                if outputs.logits is None:
                    break
                
                # 獲取下一個token的logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # 使用與generate()完全相同的生成邏輯
                # 只對新生成的部分應用重複懲罰
                generated_tokens_only = generated_ids[:, input_length:] if generated_ids.shape[1] > input_length else torch.empty((batch_size, 0), device=device, dtype=torch.long)
                
                next_token = self._apply_generation_config(
                    next_token_logits, generated_tokens_only,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample
                )
                
                next_token = next_token.unsqueeze(-1)
                
                # 更新生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                tokens_generated += 1
                
                # 更新注意力掩碼
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)
                
                # 更新緩存
                past_key_values = outputs.past_key_values
                
                # 檢查停止條件
                stop_reason = None
                should_stop = False
                
                # 檢查EOS token
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    should_stop = True
                    stop_reason = "eos_token"
                
                # 檢查最大長度
                elif tokens_generated >= max_new_tokens:
                    should_stop = True
                    stop_reason = "max_length"
                
                # 檢查最小長度
                if tokens_generated < min_new_tokens:
                    should_stop = False
                
                # 返回當前狀態
                yield {
                    "token_id": next_token.item(),
                    "token_text": "",  # 需要tokenizer來解碼
                    "generated_text": "",  # 需要tokenizer來解碼完整文本
                    "finished": should_stop,
                    "stop_reason": stop_reason,
                    "tokens_generated": tokens_generated,
                    "input_length": input_length,
                    "generated_ids": generated_ids.clone()  # 提供完整的生成序列
                }
                
                if should_stop:
                    break
                    
            except Exception as e:
                logger.error(f"流式生成過程中出錯: {e}")
                yield {
                    "token_id": None,
                    "token_text": "",
                    "generated_text": "",
                    "finished": True,
                    "stop_reason": "error",
                    "tokens_generated": tokens_generated,
                    "input_length": input_length,
                    "error": str(e)
                }
                break
        
        # 確保最終狀態被標記為完成
        if tokens_generated > 0:
            yield {
                "token_id": None,
                "token_text": "",
                "generated_text": "",
                "finished": True,
                "stop_reason": stop_reason or "completed",
                "tokens_generated": tokens_generated,
                "input_length": input_length,
                "generated_ids": generated_ids
            }

    def chat_stream(
        self,
        tokenizer,
        query: str,
        history: Optional[ConversationHistory] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        對話流式生成 - 修復版
        """
        # 確保模型處於評估模式
        self.eval()
        
        # 初始化對話歷史
        if history is None:
            history = ConversationHistory()
        
        # 添加用戶輸入到歷史
        history.add_turn("user", query)
        
        # 構建對話文本
        system_msg = system_message or getattr(self.config, 'system_message', "")
        conversation_text = ""
        
        if system_msg:
            conversation_text += f"系統：{system_msg}\n"
        
        conversation_text += history.get_conversation_text()
        conversation_text += "\n長門櫻："
        
        # 編碼輸入
        try:
            input_ids = tokenizer.encode(
                conversation_text, 
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_position_embeddings - max_new_tokens
            ).to(next(self.parameters()).device)
            
            # 創建注意力掩碼
            attention_mask = torch.ones_like(input_ids)
            
        except Exception as e:
            yield {
                "delta": "",
                "response": "",
                "finished": True,
                "error": f"輸入編碼失敗: {e}"
            }
            return
        
        # 初始化響應
        full_response = ""
        
        # 流式生成
        for output in self.stream_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs
        ):
            if output.get("finished", False):
                # 生成完成，添加到歷史
                if full_response.strip():
                    history.add_turn("assistant", full_response.strip())
                
                yield {
                    "delta": "",
                    "response": full_response,
                    "finished": True,
                    "stop_reason": output.get("stop_reason"),
                    "history": history,
                    "tokens_generated": output.get("tokens_generated", 0)
                }
                break
            
            # 解碼新token
            if "generated_ids" in output and output["generated_ids"] is not None:
                try:
                    # 只解碼新生成的部分
                    new_tokens = output["generated_ids"][:, input_ids.shape[1]:]
                    if new_tokens.shape[1] > 0:
                        new_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                        
                        # 計算增量
                        if len(new_text) > len(full_response):
                            delta = new_text[len(full_response):]
                            full_response = new_text
                        else:
                            delta = ""
                        
                        yield {
                            "delta": delta,
                            "response": full_response,
                            "finished": False,
                            "tokens_generated": output.get("tokens_generated", 0)
                        }
                    
                except Exception as e:
                    logger.warning(f"解碼token時出錯: {e}")
                    continue

# 輔助函數
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, 
    past_key_values_length: int = 0
) -> torch.Tensor:
    """創建因果掩碼"""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), 
            mask
        ], dim=-1)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, 
    past_key_values_length: int = 0
) -> torch.Tensor:
    """擴展注意力掩碼"""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    src_len = src_len + past_key_values_length

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len - past_key_values_length).to(dtype)

    if past_key_values_length > 0:
        past_mask = torch.ones((bsz, 1, tgt_len, past_key_values_length), dtype=dtype, device=mask.device)
        expanded_mask = torch.cat([past_mask, expanded_mask], dim=-1)

    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)