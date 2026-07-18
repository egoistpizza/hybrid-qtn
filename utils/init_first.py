"""
    This script is meant to be called before all Python scripts per each Python launch
"""

def init():
    
    disable_albumentations_update_checks()
    init_torch_cache()
    init_logger_basicconfig()
    
# }}}

def disable_albumentations_update_checks(): # {{{
    import os
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
# }}}

# Let the compiler cache be saved for faster loads in subsequent runs
# FIXME: Not yet verified to be working
def init_torch_cache(): # {{{
    import os
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./cache/torch cache"
    print(f"\x1b[38;5;240m[debug] Torch cache will be saved to: {os.environ["TORCHINDUCTOR_CACHE_DIR"]}\x1b[0m")
# }}}

def init_logger_basicconfig(): # {{{
    import logging
    
    logging.basicConfig(
        # filename=...,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        
        # Otherwise we get "--- Logging error ---" / "UnicodeEncodeError: 'charmap' codec can't encode character ... in
        # position ...: character maps to <undefined>"
        encoding='utf-8'
    )
    
# }}}

