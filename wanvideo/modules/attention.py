import torch
from ...utils import log

from comfy.ldm.modules.attention import optimized_attention

def attention_func_error(*args, **kwargs):
    raise ImportError("Selected attention mode not available. Please ensure required packages are installed correctly.")

from .attention_flash import flash_attention

# Import sageattn3 early so it's available in sageattn_func
try:
    from sageattn3 import sageattn3_blackwell as sageattn_blackwell
except:
    try:
        from sageattn import sageattn_blackwell
    except:
        sageattn_blackwell = None

# Helper function to get the correct pv_accum_dtype for the current GPU architecture
def _get_pv_accum_dtype():
    """Get the correct pv_accum_dtype parameter for sageattention based on GPU architecture.
    SM 120 (Blackwell) requires 'fp32+fp32' instead of the default 'fp32+fp16'."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch_code = major * 10 + minor
        # SM 120 (Blackwell) requires fp32+fp32 for fp8 kernels
        if arch_code >= 120:
            return "fp32+fp32"
    return None  # Use default for other architectures

# Sage Attention imports
# using custom ops to avoid graph breaks with torch.compile
try:
    from sageattention import sageattn
    from sageattention.core import sageattn_qk_int8_pv_fp8_cuda

    @torch.library.custom_op("wanvideo::sageattn", mutates_args=())
    def sageattn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, tensor_layout: str = "HND"
    ) -> torch.Tensor:
        # For SM 120+, use sageattn3 instead of sageattention as it may not have kernels compiled
        # Import sageattn3 here to ensure it's available in the function scope
        try:
            from sageattn3 import sageattn3_blackwell as sageattn_blackwell_func
        except:
            try:
                from sageattn import sageattn_blackwell as sageattn_blackwell_func
            except:
                sageattn_blackwell_func = None
        
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch_code = major * 10 + minor
            if arch_code >= 120:
                # Use sageattn3 for SM 120+ (Blackwell) - REQUIRED, do not fall back to sageattention
                if sageattn_blackwell_func is not None:
                    try:
                        # sageattn3_blackwell expects NHD format: (batch, seq_len, heads, head_dim)
                        # Input q, k, v are HND format: (batch, heads, seq_len, head_dim)
                        # Always convert to NHD format, matching the pattern in line 285 (sageattn_3 mode)
                        q_nhd = q.transpose(1, 2)
                        k_nhd = k.transpose(1, 2)
                        v_nhd = v.transpose(1, 2)
                        
                        result = sageattn_blackwell_func(q_nhd, k_nhd, v_nhd, per_block_mean=False)
                        
                        # Convert back to HND format
                        result = result.transpose(1, 2)
                        return result
                    except Exception as e:
                        log.error(f"Failed to use sageattn3 in sageattn_func on SM 120+: {str(e)}")
                        import traceback
                        log.error(traceback.format_exc())
                        raise RuntimeError(f"SM 120+ requires sageattn3 but it failed: {str(e)}") from e
                else:
                    log.error("SM 120+ requires sageattn3 but it is not available")
                    raise RuntimeError("SM 120+ (Blackwell) requires sageattn3. Please install and compile sageattn3.")
        
        # Get the correct pv_accum_dtype for the current GPU architecture
        pv_accum_dtype = _get_pv_accum_dtype()
        
        # For SM 89 (Ada), directly call the underlying function to bypass hardcoded pv_accum_dtype in sageattn
        # Note: SM 120+ should have been handled above
        if pv_accum_dtype is not None:
            # Calculate sm_scale if needed (head_dim^-0.5)
            head_dim = q.shape[-1]
            sm_scale = (head_dim ** -0.5) if head_dim > 0 else None
            
            # Handle dtype conversions
            if not (q.dtype == k.dtype == v.dtype):
                q_work = q
                k_work = k.to(q.dtype)
                v_work = v.to(q.dtype)
            elif q.dtype == torch.float32:
                q_work = q.to(torch.float16)
                k_work = k.to(torch.float16)
                v_work = v.to(torch.float16)
                convert_back = True
            else:
                q_work = q
                k_work = k
                v_work = v
                convert_back = False
            
            # Directly call the underlying CUDA function with correct pv_accum_dtype
            result = sageattn_qk_int8_pv_fp8_cuda(
                q_work, k_work, v_work,
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                qk_quant_gran="per_warp",
                sm_scale=sm_scale,
                return_lse=False,
                pv_accum_dtype=pv_accum_dtype
            )
            
            if convert_back:
                result = result.to(torch.float32)
            return result
        else:
            # For other architectures, use the standard sageattn function
            if not (q.dtype == k.dtype == v.dtype):
                return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
            elif q.dtype == torch.float32:
                return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
            else:
                return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)

    @sageattn_func.register_fake
    def _(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, tensor_layout="HND"):
        # Return tensor with same shape as q
        return q.clone()

    sageattn_func = torch.ops.wanvideo.sageattn

    def sageattn_func_compiled(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        # For SM 120+, use sageattn3 instead of sageattention as it may not have kernels compiled
        # Import sageattn3 here to ensure it's available in the function scope
        try:
            from sageattn3 import sageattn3_blackwell as sageattn_blackwell_func
        except:
            try:
                from sageattn import sageattn_blackwell as sageattn_blackwell_func
            except:
                sageattn_blackwell_func = None
        
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch_code = major * 10 + minor
            if arch_code >= 120:
                # Use sageattn3 for SM 120+ (Blackwell) - REQUIRED, do not fall back to sageattention
                if sageattn_blackwell_func is not None:
                    try:
                        # sageattn3_blackwell expects NHD format: (batch, seq_len, heads, head_dim)
                        # Input q, k, v are HND format: (batch, heads, seq_len, head_dim)
                        # Always convert to NHD format, matching the pattern in line 285 (sageattn_3 mode)
                        q_nhd = q.transpose(1, 2)
                        k_nhd = k.transpose(1, 2)
                        v_nhd = v.transpose(1, 2)
                        
                        result = sageattn_blackwell_func(q_nhd, k_nhd, v_nhd, per_block_mean=False)
                        
                        # Convert back to HND format
                        result = result.transpose(1, 2)
                        return result
                    except Exception as e:
                        log.error(f"Failed to use sageattn3 in sageattn_func_compiled on SM 120+: {str(e)}")
                        import traceback
                        log.error(traceback.format_exc())
                        raise RuntimeError(f"SM 120+ requires sageattn3 but it failed: {str(e)}") from e
                else:
                    log.error("SM 120+ requires sageattn3 but it is not available")
                    raise RuntimeError("SM 120+ (Blackwell) requires sageattn3. Please install and compile sageattn3.")
        
        # Get the correct pv_accum_dtype for the current GPU architecture
        pv_accum_dtype = _get_pv_accum_dtype()
        
        # For SM 89 (Ada), directly call the underlying function to bypass hardcoded pv_accum_dtype in sageattn
        # Note: SM 120+ should have been handled above
        if pv_accum_dtype is not None:
            # Calculate sm_scale if needed (head_dim^-0.5)
            head_dim = q.shape[-1]
            sm_scale = (head_dim ** -0.5) if head_dim > 0 else None
            
            # Handle dtype conversions
            if not (q.dtype == k.dtype == v.dtype):
                q_work = q
                k_work = k.to(q.dtype)
                v_work = v.to(q.dtype)
            elif q.dtype == torch.float32:
                q_work = q.to(torch.float16)
                k_work = k.to(torch.float16)
                v_work = v.to(torch.float16)
                convert_back = True
            else:
                q_work = q
                k_work = k
                v_work = v
                convert_back = False
            
            # Directly call the underlying CUDA function with correct pv_accum_dtype
            result = sageattn_qk_int8_pv_fp8_cuda(
                q_work, k_work, v_work,
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                qk_quant_gran="per_warp",
                sm_scale=sm_scale,
                return_lse=False,
                pv_accum_dtype=pv_accum_dtype
            )
            
            if convert_back:
                result = result.to(torch.float32)
            return result
        else:
            # For other architectures, use the standard sageattn function
            if not (q.dtype == k.dtype == v.dtype):
                return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
            elif q.dtype == torch.float32:
                return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
            else:
                return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
except Exception as e:
    log.warning(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        log.warning("sageattention package is not installed, sageattention will not be available")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        log.warning("sageattention DLL loading error, sageattention will not be available")
    sageattn_func = attention_func_error

try:
    from sageattention import sageattn_varlen
    from typing import List

    @torch.library.custom_op("wanvideo::sageattn_varlen", mutates_args=())
    def sageattn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lens: List[int], k_lens: List[int], max_seqlen_q: int, max_seqlen_k: int, dropout_p: float = 0.0, is_causal: bool = False) -> torch.Tensor:
        cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), dim=0)), device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_lens), dim=0)), device=q.device, dtype=torch.int32)
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn_varlen(q, k.to(q.dtype), v.to(q.dtype), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)
        elif q.dtype == torch.float32:
            return sageattn_varlen(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
        else:
            return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)

    @sageattn_varlen_func.register_fake
    def _(q, k, v, q_lens, k_lens, max_seqlen_q, max_seqlen_k, dropout_p=0.0, is_causal=False):
        # Return tensor with same shape as q
        return q.clone()
    sageattn_varlen_func = torch.ops.wanvideo.sageattn_varlen
except:
    sageattn_varlen_func = attention_func_error

# sage3 - already imported at the top, but keep this for backward compatibility
# The import is now at the top of the file to make it available in sageattn_func

try:
    from ...ultravico.sageattn.core import sage_attention as sageattn_ultravico
    @torch.library.custom_op("wanvideo::sageattn_ultravico", mutates_args=())
    def sageattn_func_ultravico(qkv: List[torch.Tensor], attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, multi_factor: float = 0.9
    ) -> torch.Tensor:
        return sageattn_ultravico(qkv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, multi_factor=multi_factor)

    @sageattn_func_ultravico.register_fake
    def _(qkv, attn_mask=None, dropout_p=0.0, is_causal=False, multi_factor=0.9):
        return torch.empty_like(qkv[0]).contiguous()
    sageattn_func_ultravico = torch.ops.wanvideo.sageattn_ultravico
except:
    sageattn_func_ultravico = attention_func_error

def attention(q, k, v, q_lens=None, k_lens=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.,
    softmax_scale=None, q_scale=None, causal=False,  window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
    attention_mode='sdpa', attn_mask=None, multi_factor=0.9, heads=128):
    if "flash" in attention_mode:
        return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype, version=2 if attention_mode == 'flash_attn_2' else 3,
        )
    elif attention_mode == 'sageattn_3':
        return sageattn_blackwell(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), per_block_mean=False).transpose(1,2).contiguous()
    elif attention_mode == 'sageattn_varlen':
        return sageattn_varlen_func(q,k,v, q_lens=q_lens, k_lens=k_lens, max_seqlen_k=max_seqlen_k, max_seqlen_q=max_seqlen_q)
    elif attention_mode == 'sageattn_compiled': # for sage versions that allow torch.compile, may be redundant now as other sageattn ops are wrapper in custom ops
        return sageattn_func_compiled(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn':
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn_ultravico':
        return sageattn_func_ultravico([q, k, v], multi_factor=multi_factor).contiguous()
    elif attention_mode == 'comfy':
        return optimized_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), heads=heads, skip_reshape=True)
    else: # sdpa
        if not (q.dtype == k.dtype == v.dtype):
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype), attn_mask=attn_mask).transpose(1, 2).contiguous()
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(1, 2).contiguous()
