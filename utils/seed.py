def seed_everything(seed: int = 42) -> None: # {{{
    """Fix random seeds for reproducibility."""
    
    import random
    import os
    import numpy as np
    import torch
    import logging
    
    logger = logging.getLogger(__name__)
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seed set to {seed}")
    
    # TODO: Call logger for these!
    print(f"\x1b[38;5;240m[debug] random.seed({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] os.environ['PYTHONHASHSEED'] = str({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] np.random.seed({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] torch.manual_seed({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] torch.cuda.manual_seed({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] torch.cuda.manual_seed_all({seed})\x1b[0m")
    print(f"\x1b[38;5;240m[debug] torch.backends.cudnn.deterministic = True\x1b[0m")
    print(f"\x1b[38;5;240m[debug] torch.backends.cudnn.benchmark = True\x1b[0m")
# }}}